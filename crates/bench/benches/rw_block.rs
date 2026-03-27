//! Read-then-write benchmark at 10k TPS scale.
//!
//! Models a realistic block commit:
//!   1. Read old state from DB (must read old value to compute LtHash delta)
//!   2. Apply EVM execution results (new nonce/balance/storage values)
//!   3. Write new state + compute new state root
//!
//! Scenarios based on 10k TPS / 1s block time estimate:
//!   conservative: ~25k changes  (11.7k accounts + 13.3k slots)
//!   typical:      ~50k changes  (23.4k accounts + 26.6k slots)
//!   heavy:        ~100k changes (46.8k accounts + 53.2k slots)
//!
//! Competitors:
//!   lthash_seq  — flat RocksDB reads + sequential WorldHash update
//!   lthash_par  — flat RocksDB reads + rayon parallel WorldHash update
//!   mpt         — EthTrie<MemoryDB> incremental insert (in-memory, no DB IO — lower bound)

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use alloy_trie::EMPTY_ROOT_HASH;
use bench::generate_state;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, MemoryDB, Trie};
use lthash::{apply_parallel, apply_sequential, AccountState, StateChange, WorldHash};
use rand::{rngs::StdRng, Rng, SeedableRng};
use state_db::StateDb;
use std::sync::Arc;
use tempfile::TempDir;

// ── MPT helpers (same as state_root bench) ────────────────────────────────────

fn rlp_u256(v: U256) -> Vec<u8> {
    let mut buf = Vec::new();
    v.encode(&mut buf);
    buf
}

fn rlp_account(nonce: u64, balance: U256, storage_root: B256, code_hash: B256) -> Vec<u8> {
    let mut buf = Vec::new();
    let payload = nonce.length() + balance.length() + storage_root.length() + code_hash.length();
    alloy_rlp::Header { list: true, payload_length: payload }.encode(&mut buf);
    nonce.encode(&mut buf);
    balance.encode(&mut buf);
    storage_root.encode(&mut buf);
    code_hash.encode(&mut buf);
    buf
}

// ── Block delta description ───────────────────────────────────────────────────

/// One account's changes in a block.
#[derive(Clone)]
struct AccountDelta {
    addr: Address,
    /// Old state (as in the DB before this block).
    old: AccountState,
    /// New state after EVM execution.
    new: AccountState,
    /// Storage changes: (slot, old_value, new_value)
    storage: Vec<(B256, U256, U256)>,
}

// ── Pre-built state fixtures ──────────────────────────────────────────────────

struct Fixture {
    /// Temp dir keeping the RocksDB alive.
    _dir: TempDir,
    /// Pre-populated LtHash + RocksDB state.
    db: StateDb,
    /// Pre-built WorldHash of the base state (to skip re-building each iter).
    base_world: WorldHash,
    /// Pre-built EthTrie representing the same base state.
    mpt: MptFixture,
    /// The block deltas to apply.
    deltas: Vec<AccountDelta>,
}

struct MptFixture {
    account_trie: EthTrie<MemoryDB>,
    /// addr → (storage trie, current storage root)
    storage_tries: std::collections::HashMap<Address, (EthTrie<MemoryDB>, B256)>,
}

impl MptFixture {
    fn apply_block(&mut self, deltas: &[AccountDelta]) -> B256 {
        for d in deltas {
            let storage_root = if d.storage.is_empty() {
                self.storage_tries
                    .get(&d.addr)
                    .map(|(_, r)| *r)
                    .unwrap_or(EMPTY_ROOT_HASH)
            } else if let Some((strie, _)) = self.storage_tries.get_mut(&d.addr) {
                for &(slot, _old, new_val) in &d.storage {
                    let hk = keccak256(slot);
                    if new_val.is_zero() {
                        strie.remove(hk.as_slice()).unwrap();
                    } else {
                        strie.insert(hk.as_slice(), &rlp_u256(new_val)).unwrap();
                    }
                }
                strie.root_hash().unwrap()
            } else {
                EMPTY_ROOT_HASH
            };

            let hashed_addr = keccak256(d.addr);
            let encoded =
                rlp_account(d.new.nonce, d.new.balance, storage_root, d.new.code_hash);
            self.account_trie.insert(hashed_addr.as_slice(), &encoded).unwrap();
        }
        self.account_trie.root_hash().unwrap()
    }
}

// ── Fixture builder ───────────────────────────────────────────────────────────

fn build_fixture(n_accounts: usize, n_slots: usize) -> Fixture {
    const BASE_ACCOUNTS: usize = 100_000;
    const BASE_SLOTS_EACH: usize = 4;

    let base = generate_state(BASE_ACCOUNTS, BASE_SLOTS_EACH, 1);

    // ── RocksDB + LtHash ─────────────────────────────────────────────────────
    let dir = tempfile::tempdir().unwrap();
    let mut db = StateDb::open(dir.path()).unwrap();

    let init_changes: Vec<StateChange> = base
        .accounts
        .iter()
        .flat_map(|acc| {
            let mut v = vec![StateChange::Account {
                addr: acc.addr,
                old: None,
                new: AccountState {
                    nonce: acc.nonce,
                    balance: acc.balance,
                    code_hash: acc.code_hash,
                },
            }];
            for &(slot, val) in &acc.storage {
                v.push(StateChange::Storage {
                    addr: acc.addr,
                    slot,
                    old_value: U256::ZERO,
                    new_value: val,
                });
            }
            v
        })
        .collect();

    db.apply_parallel(&init_changes).unwrap();
    let base_world = {
        let mut w = WorldHash::new();
        lthash::apply_sequential(&mut w, &init_changes);
        w
    };

    // ── EthTrie (in-memory MPT) ───────────────────────────────────────────────
    let mpt_db = Arc::new(MemoryDB::new(true));
    let mut account_trie = EthTrie::new(Arc::clone(&mpt_db));
    let mut storage_tries = std::collections::HashMap::new();

    for acc in &base.accounts {
        let sdb = Arc::new(MemoryDB::new(true));
        let mut strie = EthTrie::new(Arc::clone(&sdb));
        for &(slot, val) in &acc.storage {
            if !val.is_zero() {
                strie.insert(keccak256(slot).as_slice(), &rlp_u256(val)).unwrap();
            }
        }
        let storage_root = strie.root_hash().unwrap();
        storage_tries.insert(acc.addr, (strie, storage_root));

        let hashed_addr = keccak256(acc.addr);
        let encoded =
            rlp_account(acc.nonce, acc.balance, storage_root, acc.code_hash);
        account_trie.insert(hashed_addr.as_slice(), &encoded).unwrap();
    }
    account_trie.root_hash().unwrap();

    // ── Block delta ───────────────────────────────────────────────────────────
    let mut rng = StdRng::seed_from_u64(42);
    let slots_each = (n_slots / n_accounts).max(1);

    let deltas: Vec<AccountDelta> = base.accounts[..n_accounts]
        .iter()
        .map(|acc| {
            let old = AccountState {
                nonce: acc.nonce,
                balance: acc.balance,
                code_hash: acc.code_hash,
            };
            let new = AccountState {
                nonce: acc.nonce + 1,
                balance: acc.balance + U256::from(rng.gen::<u64>()),
                code_hash: acc.code_hash,
            };
            let storage: Vec<_> = acc
                .storage
                .iter()
                .take(slots_each)
                .map(|&(slot, old_val)| (slot, old_val, U256::from(rng.gen::<u128>())))
                .collect();
            AccountDelta { addr: acc.addr, old, new, storage }
        })
        .collect();

    Fixture {
        _dir: dir,
        db,
        base_world,
        mpt: MptFixture { account_trie, storage_tries },
        deltas,
    }
}

// ── LtHash read-then-write ────────────────────────────────────────────────────

fn lthash_rtw_sequential(db: &mut StateDb, base_world: &WorldHash, deltas: &[AccountDelta]) -> B256 {
    // 1. Read: batch fetch old account state from RocksDB
    let addrs: Vec<Address> = deltas.iter().map(|d| d.addr).collect();
    let _old_accounts = db.multi_get_accounts(&addrs).unwrap();

    let storage_keys: Vec<(Address, B256)> = deltas
        .iter()
        .flat_map(|d| d.storage.iter().map(|&(slot, _, _)| (d.addr, slot)))
        .collect();
    let _old_storage = db.multi_get_storage(&storage_keys).unwrap();
    // (In real EVM: old values come from revm's pre-state cache, which
    //  was populated via these same DB reads during execution. We model
    //  the read cost here; the old values are already in `deltas`.)

    // 2. Compute state changes with old+new values
    let changes: Vec<StateChange> = deltas
        .iter()
        .flat_map(|d| {
            let mut v = vec![StateChange::Account {
                addr: d.addr,
                old: Some(d.old),
                new: d.new,
            }];
            for &(slot, old_val, new_val) in &d.storage {
                v.push(StateChange::Storage {
                    addr: d.addr,
                    slot,
                    old_value: old_val,
                    new_value: new_val,
                });
            }
            v
        })
        .collect();

    // 3. Update WorldHash (sequential)
    let mut world = base_world.clone();
    apply_sequential(&mut world, &changes);

    // 4. Write batch to RocksDB + persist world hash
    db.apply(&changes).unwrap();

    world.state_root()
}

fn lthash_rtw_parallel(db: &mut StateDb, base_world: &WorldHash, deltas: &[AccountDelta]) -> B256 {
    // Same read phase
    let addrs: Vec<Address> = deltas.iter().map(|d| d.addr).collect();
    let _old_accounts = db.multi_get_accounts(&addrs).unwrap();
    let storage_keys: Vec<(Address, B256)> = deltas
        .iter()
        .flat_map(|d| d.storage.iter().map(|&(slot, _, _)| (d.addr, slot)))
        .collect();
    let _old_storage = db.multi_get_storage(&storage_keys).unwrap();

    let changes: Vec<StateChange> = deltas
        .iter()
        .flat_map(|d| {
            let mut v = vec![StateChange::Account {
                addr: d.addr,
                old: Some(d.old),
                new: d.new,
            }];
            for &(slot, old_val, new_val) in &d.storage {
                v.push(StateChange::Storage {
                    addr: d.addr,
                    slot,
                    old_value: old_val,
                    new_value: new_val,
                });
            }
            v
        })
        .collect();

    // Parallel WorldHash update
    let mut world = base_world.clone();
    apply_parallel(&mut world, &changes);

    db.apply_parallel(&changes).unwrap();
    world.state_root()
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_rw(c: &mut Criterion) {
    // Scenarios: (label, n_accounts, n_storage_slots)
    let scenarios: &[(&str, usize, usize)] = &[
        ("conservative_25k", 11_700, 13_300),
        ("typical_50k",      23_400, 26_600),
        ("heavy_100k",       46_800, 53_200),
    ];

    let mut group = c.benchmark_group("rw_block_commit");

    for &(label, n_acc, n_slots) in scenarios {
        println!("Building fixture: {label} ({n_acc} accounts, {n_slots} slots)…");
        let mut fixture = build_fixture(n_acc, n_slots);
        let total = n_acc + n_slots;
        group.throughput(Throughput::Elements(total as u64));

        // ── LtHash sequential (read + hash + write) ──────────────────────────
        group.bench_function(BenchmarkId::new("lthash_seq", label), |b| {
            b.iter(|| {
                lthash_rtw_sequential(&mut fixture.db, &fixture.base_world, &fixture.deltas)
            });
        });

        // ── LtHash parallel (read + parallel hash + write) ───────────────────
        group.bench_function(BenchmarkId::new("lthash_par", label), |b| {
            b.iter(|| {
                lthash_rtw_parallel(&mut fixture.db, &fixture.base_world, &fixture.deltas)
            });
        });

        // ── MPT in-memory (no DB IO — lower bound for MPT) ───────────────────
        group.bench_function(BenchmarkId::new("mpt_mem", label), |b| {
            b.iter(|| fixture.mpt.apply_block(&fixture.deltas));
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(8));
    targets = bench_rw
}
criterion_main!(benches);
