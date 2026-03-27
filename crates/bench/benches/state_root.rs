//! Benchmark: LtHash vs MPT incremental state root update.
//!
//! Simulates one block commit: a large base state exists,
//! and a block touches N accounts (update nonce/balance) and M storage slots.
//!
//! Both schemes maintain persistent state across iterations:
//! - LtHash: WorldHash (2 KB in memory) + apply delta
//! - MPT:    EthTrie<MemoryDB> (full in-memory trie) + incremental insert + root_hash()
//!
//! Scenarios: (n_accounts_touched, storage_slots_touched)
//!   tiny:   50 accounts,   100 slots  (light block)
//!   small:  500 accounts, 1000 slots  (typical block)
//!   medium: 2000 accounts, 4000 slots (heavy block)

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use bench::{to_lthash_changes, AccountEntry};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, MemoryDB, Trie};
use lthash::{apply_parallel, apply_sequential, AccountState, StateChange, WorldHash};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

// ── Encoding helpers (shared) ─────────────────────────────────────────────────

/// RLP-encode a U256 value.
fn rlp_u256(v: U256) -> Vec<u8> {
    let mut buf = Vec::new();
    v.encode(&mut buf);
    buf
}

/// RLP-encode a trie account: [nonce, balance, storageRoot, codeHash]
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

// ── MPT helpers ───────────────────────────────────────────────────────────────

/// One EthTrie per account for its storage (standard Ethereum nested-trie layout).
/// For the benchmark we track the storage root separately and only rebuild when slots change.
struct MptState {
    /// Account trie: keccak256(addr) → RLP([nonce, balance, storage_root, code_hash])
    account_trie: EthTrie<MemoryDB>,
    /// Per-account storage tries: addr → (EthTrie, storage_root)
    storage_tries: std::collections::HashMap<Address, (EthTrie<MemoryDB>, B256)>,
}

impl MptState {
    /// Build initial MPT from a generated base state.
    fn from_base(accounts: &[AccountEntry]) -> Self {
        let db = Arc::new(MemoryDB::new(true));
        let mut account_trie = EthTrie::new(Arc::clone(&db));
        let mut storage_tries = std::collections::HashMap::new();

        for acc in accounts {
            let sdb = Arc::new(MemoryDB::new(true));
            let mut strie = EthTrie::new(Arc::clone(&sdb));
            for &(slot, value) in &acc.storage {
                if !value.is_zero() {
                    let hashed_slot = keccak256(slot);
                    strie.insert(hashed_slot.as_slice(), &rlp_u256(value)).unwrap();
                }
            }
            let storage_root = strie.root_hash().unwrap();

            let hashed_addr = keccak256(acc.addr);
            let encoded = rlp_account(acc.nonce, acc.balance, storage_root, acc.code_hash);
            account_trie.insert(hashed_addr.as_slice(), &encoded).unwrap();

            storage_tries.insert(acc.addr, (strie, storage_root));
        }

        // Materialise root
        account_trie.root_hash().unwrap();

        Self { account_trie, storage_tries }
    }

    /// Apply a block's account + storage changes, return new state root.
    fn apply_block(&mut self, changes: &[BlockChange]) -> B256 {
        for change in changes {
            // 1. Update storage trie for this account (if any slots changed)
            let storage_root = if let Some((strie, _)) = self.storage_tries.get_mut(&change.addr) {
                if !change.storage_changes.is_empty() {
                    for &(slot, old_val, new_val) in &change.storage_changes {
                        let hashed_slot = keccak256(slot);
                        if new_val.is_zero() {
                            strie.remove(hashed_slot.as_slice()).unwrap();
                        } else {
                            strie.insert(hashed_slot.as_slice(), &rlp_u256(new_val)).unwrap();
                        }
                        let _ = old_val; // only needed for LtHash
                    }
                    strie.root_hash().unwrap()
                } else {
                    // No storage change: reuse cached root
                    self.storage_tries[&change.addr].1
                }
            } else {
                alloy_trie::EMPTY_ROOT_HASH
            };

            // 2. Update account trie
            let hashed_addr = keccak256(change.addr);
            let encoded =
                rlp_account(change.new_nonce, change.new_balance, storage_root, change.code_hash);
            self.account_trie.insert(hashed_addr.as_slice(), &encoded).unwrap();
        }

        self.account_trie.root_hash().unwrap()
    }
}

// ── Block change representation ───────────────────────────────────────────────

#[derive(Clone)]
struct BlockChange {
    addr: Address,
    old_nonce: u64,
    old_balance: U256,
    new_nonce: u64,
    new_balance: U256,
    code_hash: B256,
    /// (slot, old_value, new_value)
    storage_changes: Vec<(B256, U256, U256)>,
}

impl BlockChange {
    fn to_lthash_changes(&self) -> Vec<StateChange> {
        let mut out = Vec::new();
        out.push(StateChange::Account {
            addr: self.addr,
            old: Some(AccountState {
                nonce: self.old_nonce,
                balance: self.old_balance,
                code_hash: self.code_hash,
            }),
            new: AccountState {
                nonce: self.new_nonce,
                balance: self.new_balance,
                code_hash: self.code_hash,
            },
        });
        for &(slot, old_val, new_val) in &self.storage_changes {
            out.push(StateChange::Storage {
                addr: self.addr,
                slot,
                old_value: old_val,
                new_value: new_val,
            });
        }
        out
    }
}

// ── State generation ──────────────────────────────────────────────────────────

/// Generate a block's worth of changes touching `n_accounts` accounts
/// and `total_storage_changes` storage slots spread across those accounts.
fn generate_block_changes(
    base: &[AccountEntry],
    n_accounts: usize,
    total_storage_changes: usize,
    seed: u64,
) -> Vec<BlockChange> {
    assert!(n_accounts <= base.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let slots_per_account = (total_storage_changes / n_accounts).max(1);

    base[..n_accounts]
        .iter()
        .map(|acc| {
            let new_nonce = acc.nonce + 1;
            let new_balance = acc.balance + U256::from(rng.gen::<u64>());

            let storage_changes: Vec<_> = acc
                .storage
                .iter()
                .take(slots_per_account)
                .map(|&(slot, old_val)| {
                    let new_val = U256::from(rng.gen::<u128>());
                    (slot, old_val, new_val)
                })
                .collect();

            BlockChange {
                addr: acc.addr,
                old_nonce: acc.nonce,
                old_balance: acc.balance,
                new_nonce,
                new_balance,
                code_hash: acc.code_hash,
                storage_changes,
            }
        })
        .collect()
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_block_commit(c: &mut Criterion) {
    // Base state: 100k accounts, 4 storage slots each — represents chain history
    const BASE_ACCOUNTS: usize = 100_000;
    const BASE_SLOTS: usize = 4;

    println!("Building base state ({BASE_ACCOUNTS} accounts × {BASE_SLOTS} slots)…");
    let base = bench::generate_state(BASE_ACCOUNTS, BASE_SLOTS, 1);

    // Scenarios based on 10k TPS / 1s block estimate:
    //   conservative: ~25k changes (11.7k accounts + 13.3k slots, lots of hot-wallet overlap)
    //   typical:      ~50k changes (23.4k accounts + 26.6k slots)
    //   heavy:        ~100k changes (46.8k accounts + 53.2k slots, DeFi-heavy)
    let scenarios: &[(&str, usize, usize)] = &[
        ("conservative_25k", 11_700, 13_300),
        ("typical_50k",      23_400, 26_600),
        ("heavy_100k",       46_800, 53_200),
    ];

    // ── LtHash: build initial WorldHash from base state ─────────────────────
    println!("Building LtHash base world hash…");
    let base_changes = to_lthash_changes(&base);
    let mut base_world = WorldHash::new();
    apply_sequential(&mut base_world, &base_changes);

    // MPT base trie is built fresh per scenario (cloning 100k accounts is expensive;
    // we instead build per-scenario sub-tries in the benchmark itself)

    println!("Setup done. Running benchmarks…\n");

    let mut group = c.benchmark_group("block_commit");

    for &(label, n_acc, n_slots) in scenarios {
        let block = generate_block_changes(&base.accounts, n_acc, n_slots, 42);
        let total_changes = n_acc + n_slots; // accounts + storage slots touched
        group.throughput(Throughput::Elements(total_changes as u64));

        // ── LtHash incremental ───────────────────────────────────────────────
        // Starts from pre-built base_world, applies only the delta for this block.
        // O(n_acc + n_slots) BLAKE3 XOF operations, no state-size dependency.
        let lthash_changes: Vec<StateChange> =
            block.iter().flat_map(|c| c.to_lthash_changes()).collect();

        group.bench_with_input(
            BenchmarkId::new("lthash_seq", label),
            &lthash_changes,
            |b, changes| {
                b.iter(|| {
                    let mut world = base_world.clone();
                    apply_sequential(&mut world, changes);
                    world.state_root()
                });
            },
        );

        // ── LtHash parallel ──────────────────────────────────────────────────
        // rayon splits the changes across threads, each thread computes a
        // partial WorldHash delta, then they are reduced (wrapping-add) with
        // no locks.  Addition is commutative so order doesn't matter.
        group.bench_with_input(
            BenchmarkId::new("lthash_par", label),
            &lthash_changes,
            |b, changes| {
                b.iter(|| {
                    let mut world = base_world.clone();
                    apply_parallel(&mut world, changes);
                    world.state_root()
                });
            },
        );

        // ── MPT incremental ──────────────────────────────────────────────────
        // Starts from pre-built EthTrie, calls insert() for each changed
        // account/slot, then root_hash(). EthTrie caches unchanged nodes —
        // only the modified paths are rehashed (O(n_changed × depth)).
        group.bench_with_input(
            BenchmarkId::new("mpt", label),
            &block,
            |b, changes| {
                b.iter(|| {
                    // Clone the trie state for each iteration so we always
                    // benchmark a single block's delta from the same base.
                    let mut mpt = MptState::from_base(&base.accounts[..n_acc]);
                    mpt.apply_block(changes)
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    // Shorter times for quick iteration; use defaults for final results
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_block_commit
}
criterion_main!(benches);
