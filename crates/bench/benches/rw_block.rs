//! Read-then-write benchmark at 10k TPS scale.
//!
//! Both competitors use RocksDB for persistence — apples-to-apples:
//!
//!   lthash_par  — flat KV reads  (1 get/entry)  + parallel BLAKE3 + batch write
//!   mpt_rocksdb — trie node reads (O(depth)/entry) + keccak256 path + node writes
//!
//! Scenarios based on 10k TPS / 1s block time:
//!   conservative: ~25k changes  (11.7k accounts + 13.3k slots)
//!   typical:      ~50k changes  (23.4k accounts + 26.6k slots)
//!   heavy:        ~100k changes (46.8k accounts + 53.2k slots)

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use alloy_trie::EMPTY_ROOT_HASH;
use bench::generate_state;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, Trie, DB as TrieDB};
use lthash::{apply_parallel, AccountState, StateChange, WorldHash};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use state_db::StateDb;
use std::{path::Path, sync::Arc};
use tempfile::TempDir;

// ── RocksDB backend for EthTrie ───────────────────────────────────────────────
//
// eth_trie stores nodes keyed by their hash (32B). Each EthTrie instance
// gets its own prefix so account trie and per-account storage tries can
// coexist in a single RocksDB column family.

struct PrefixedRocksDb {
    db: Arc<DB>,
    /// Column family for MPT trie nodes.
    cf_name: &'static str,
    /// Key prefix: distinguishes different trie instances.
    prefix: Vec<u8>,
}

impl PrefixedRocksDb {
    fn prefixed(&self, key: &[u8]) -> Vec<u8> {
        let mut k = Vec::with_capacity(self.prefix.len() + key.len());
        k.extend_from_slice(&self.prefix);
        k.extend_from_slice(key);
        k
    }

    fn cf(&self) -> &rocksdb::ColumnFamily {
        self.db.cf_handle(self.cf_name).unwrap()
    }
}

impl TrieDB for PrefixedRocksDb {
    type Error = rocksdb::Error;

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        self.db.get_cf(self.cf(), self.prefixed(key))
    }

    fn insert(&self, key: &[u8], value: Vec<u8>) -> Result<(), Self::Error> {
        self.db.put_cf(self.cf(), self.prefixed(key), value)
    }

    fn insert_batch(&self, keys: Vec<Vec<u8>>, values: Vec<Vec<u8>>) -> Result<(), Self::Error> {
        let mut batch = WriteBatch::default();
        for (k, v) in keys.into_iter().zip(values) {
            batch.put_cf(self.cf(), self.prefixed(&k), v);
        }
        self.db.write(batch)
    }

    fn remove(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.db.delete_cf(self.cf(), self.prefixed(key))
    }

    fn remove_batch(&self, keys: &[Vec<u8>]) -> Result<(), Self::Error> {
        let mut batch = WriteBatch::default();
        for k in keys {
            batch.delete_cf(self.cf(), self.prefixed(k));
        }
        self.db.write(batch)
    }

    fn flush(&self) -> Result<(), Self::Error> {
        Ok(()) // writes already committed via put_cf / write(batch)
    }
}

// ── Shared RocksDB open helper ────────────────────────────────────────────────

const CF_MPT: &str = "mpt_nodes"; // trie nodes for EthTrie

fn open_shared_db<P: AsRef<Path>>(path: P) -> Arc<DB> {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    let cf_opts = Options::default();
    let cfs = vec![
        ColumnFamilyDescriptor::new("default", cf_opts.clone()),
        ColumnFamilyDescriptor::new(CF_MPT, cf_opts),
    ];
    Arc::new(DB::open_cf_descriptors(&opts, path, cfs).unwrap())
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

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

// Prefix for account trie: 0x00 (1 byte)
fn account_trie_prefix() -> Vec<u8> { vec![0x00] }

// Prefix for storage trie of `addr`: 0x01 || addr (21 bytes total)
fn storage_trie_prefix(addr: &Address) -> Vec<u8> {
    let mut p = vec![0x01];
    p.extend_from_slice(addr.as_slice());
    p
}

// ── MPT RocksDB state ─────────────────────────────────────────────────────────

struct MptRocksDb {
    db: Arc<DB>,
    account_trie: EthTrie<PrefixedRocksDb>,
    /// addr → (storage trie, cached storage root)
    storage_tries: std::collections::HashMap<Address, (EthTrie<PrefixedRocksDb>, B256)>,
}

impl MptRocksDb {
    fn new(db: Arc<DB>) -> Self {
        let acc_backend = Arc::new(PrefixedRocksDb {
            db: Arc::clone(&db),
            cf_name: CF_MPT,
            prefix: account_trie_prefix(),
        });
        Self {
            db,
            account_trie: EthTrie::new(acc_backend),
            storage_tries: Default::default(),
        }
    }

    fn insert_account(
        &mut self,
        addr: &Address,
        nonce: u64,
        balance: U256,
        code_hash: B256,
        storage_items: &[(B256, U256)],
    ) {
        let storage_root = if storage_items.is_empty() {
            EMPTY_ROOT_HASH
        } else {
            let strie = self.storage_tries.entry(*addr).or_insert_with(|| {
                let backend = Arc::new(PrefixedRocksDb {
                    db: Arc::clone(&self.db),
                    cf_name: CF_MPT,
                    prefix: storage_trie_prefix(addr),
                });
                (EthTrie::new(backend), EMPTY_ROOT_HASH)
            });
            for &(slot, val) in storage_items {
                if val.is_zero() {
                    strie.0.remove(keccak256(slot).as_slice()).unwrap();
                } else {
                    strie.0.insert(keccak256(slot).as_slice(), &rlp_u256(val)).unwrap();
                }
            }
            let root = strie.0.root_hash().unwrap();
            strie.1 = root;
            root
        };

        let hashed_addr = keccak256(addr);
        self.account_trie
            .insert(hashed_addr.as_slice(), &rlp_account(nonce, balance, storage_root, code_hash))
            .unwrap();
    }

    fn root_hash(&mut self) -> B256 {
        self.account_trie.root_hash().unwrap()
    }
}

// ── Block delta ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AccountDelta {
    addr: Address,
    old: AccountState,
    new: AccountState,
    storage: Vec<(B256, U256, U256)>, // (slot, old_value, new_value)
}

// ── Fixture ───────────────────────────────────────────────────────────────────

struct Fixture {
    _dir: TempDir,
    /// LtHash + flat KV (StateDb wraps its own RocksDB)
    lthash_db: StateDb,
    base_world: WorldHash,
    /// MPT + RocksDB (EthTrie with PrefixedRocksDb backend)
    mpt_db: MptRocksDb,
    deltas: Vec<AccountDelta>,
}

fn build_fixture(n_accounts: usize, n_slots: usize) -> Fixture {
    const BASE: usize = 100_000;
    const BASE_SLOTS: usize = 4;

    let base = generate_state(BASE, BASE_SLOTS, 1);
    let dir = tempfile::tempdir().unwrap();

    // ── LtHash StateDb ────────────────────────────────────────────────────────
    let lthash_path = dir.path().join("lthash");
    let mut lthash_db = StateDb::open(&lthash_path).unwrap();

    let init: Vec<StateChange> = base.accounts.iter().flat_map(|acc| {
        let mut v = vec![StateChange::Account {
            addr: acc.addr,
            old: None,
            new: AccountState { nonce: acc.nonce, balance: acc.balance, code_hash: acc.code_hash },
        }];
        for &(slot, val) in &acc.storage {
            v.push(StateChange::Storage {
                addr: acc.addr, slot, old_value: U256::ZERO, new_value: val,
            });
        }
        v
    }).collect();

    lthash_db.apply_parallel(&init).unwrap();
    let mut base_world = WorldHash::new();
    lthash::apply_sequential(&mut base_world, &init);

    // ── MPT RocksDB ───────────────────────────────────────────────────────────
    let mpt_path = dir.path().join("mpt");
    let mpt_raw_db = open_shared_db(&mpt_path);
    let mut mpt_db = MptRocksDb::new(Arc::clone(&mpt_raw_db));

    for acc in &base.accounts {
        mpt_db.insert_account(&acc.addr, acc.nonce, acc.balance, acc.code_hash, &acc.storage);
    }
    mpt_db.root_hash(); // flush pending writes to DB

    // ── Block deltas ──────────────────────────────────────────────────────────
    let mut rng = StdRng::seed_from_u64(42);
    let slots_each = (n_slots / n_accounts).max(1);

    let deltas: Vec<AccountDelta> = base.accounts[..n_accounts].iter().map(|acc| {
        AccountDelta {
            addr: acc.addr,
            old: AccountState { nonce: acc.nonce, balance: acc.balance, code_hash: acc.code_hash },
            new: AccountState {
                nonce: acc.nonce + 1,
                balance: acc.balance + U256::from(rng.gen::<u64>()),
                code_hash: acc.code_hash,
            },
            storage: acc.storage.iter().take(slots_each)
                .map(|&(slot, old)| (slot, old, U256::from(rng.gen::<u128>())))
                .collect(),
        }
    }).collect();

    Fixture { _dir: dir, lthash_db, base_world, mpt_db, deltas }
}

// ── LtHash block commit ───────────────────────────────────────────────────────
//
// 1. Batch read old values from flat RocksDB  (1 get per entry)
// 2. Parallel BLAKE3 XOF delta computation
// 3. Atomic WriteBatch to RocksDB

fn lthash_block_commit(db: &mut StateDb, base_world: &WorldHash, deltas: &[AccountDelta]) -> B256 {
    // 1. Read old state (revm pre-state cache equivalent)
    let addrs: Vec<Address> = deltas.iter().map(|d| d.addr).collect();
    let _old_accounts = db.multi_get_accounts(&addrs).unwrap();
    let storage_keys: Vec<(Address, B256)> = deltas.iter()
        .flat_map(|d| d.storage.iter().map(|&(slot, ..)| (d.addr, slot)))
        .collect();
    let _old_storage = db.multi_get_storage(&storage_keys).unwrap();

    // 2. Build changes (old values already in deltas from EVM execution)
    let changes: Vec<StateChange> = deltas.iter().flat_map(|d| {
        let mut v = vec![StateChange::Account {
            addr: d.addr, old: Some(d.old), new: d.new,
        }];
        for &(slot, old_val, new_val) in &d.storage {
            v.push(StateChange::Storage {
                addr: d.addr, slot, old_value: old_val, new_value: new_val,
            });
        }
        v
    }).collect();

    // 3. Parallel hash + atomic write
    let mut world = base_world.clone();
    apply_parallel(&mut world, &changes);
    db.apply_parallel(&changes).unwrap();
    world.state_root()
}

// ── MPT RocksDB block commit ──────────────────────────────────────────────────
//
// 1. EthTrie reads O(depth) trie nodes from RocksDB per changed entry
// 2. keccak256 path recomputation
// 3. Updated trie nodes written back to RocksDB

fn mpt_block_commit(mpt: &mut MptRocksDb, deltas: &[AccountDelta]) -> B256 {
    for d in deltas {
        let storage_root = if d.storage.is_empty() {
            mpt.storage_tries.get(&d.addr).map(|(_, r)| *r).unwrap_or(EMPTY_ROOT_HASH)
        } else {
            let entry = mpt.storage_tries.entry(d.addr).or_insert_with(|| {
                let backend = Arc::new(PrefixedRocksDb {
                    db: Arc::clone(&mpt.db),
                    cf_name: CF_MPT,
                    prefix: storage_trie_prefix(&d.addr),
                });
                (EthTrie::new(backend), EMPTY_ROOT_HASH)
            });
            for &(slot, _old, new_val) in &d.storage {
                let hk = keccak256(slot);
                if new_val.is_zero() {
                    entry.0.remove(hk.as_slice()).unwrap();
                } else {
                    entry.0.insert(hk.as_slice(), &rlp_u256(new_val)).unwrap();
                }
            }
            let root = entry.0.root_hash().unwrap();
            entry.1 = root;
            root
        };

        let hashed_addr = keccak256(d.addr);
        let encoded = rlp_account(d.new.nonce, d.new.balance, storage_root, d.new.code_hash);
        mpt.account_trie.insert(hashed_addr.as_slice(), &encoded).unwrap();
    }
    mpt.account_trie.root_hash().unwrap()
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_rw(c: &mut Criterion) {
    let scenarios: &[(&str, usize, usize)] = &[
        ("conservative_25k", 11_700, 13_300),
        ("typical_50k",      23_400, 26_600),
        ("heavy_100k",       46_800, 53_200),
    ];

    let mut group = c.benchmark_group("rw_block_commit");

    for &(label, n_acc, n_slots) in scenarios {
        println!("Building fixture: {label} ({n_acc} accounts, {n_slots} slots)…");
        let mut fix = build_fixture(n_acc, n_slots);
        group.throughput(Throughput::Elements((n_acc + n_slots) as u64));

        // LtHash: flat KV reads + parallel BLAKE3 + batch write (all RocksDB)
        group.bench_function(BenchmarkId::new("lthash_par", label), |b| {
            b.iter(|| lthash_block_commit(&mut fix.lthash_db, &fix.base_world, &fix.deltas));
        });

        // MPT: trie node reads + keccak256 path updates + node writes (all RocksDB)
        group.bench_function(BenchmarkId::new("mpt_rocksdb", label), |b| {
            b.iter(|| mpt_block_commit(&mut fix.mpt_db, &fix.deltas));
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
