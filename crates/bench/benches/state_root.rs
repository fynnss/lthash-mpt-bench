//! Benchmark: LtHash vs MPT (serial + 16-way parallel) incremental state root update.
//!
//! Simulates one block commit against a 100k-account base state.
//!
//! Competitors:
//!   lthash_par — O(N) BLAKE3 XOF, fully parallel (rayon), no structural dependency
//!   mpt        — EthTrie<MemoryDB> serial incremental (O(N × depth) keccak256)
//!   mpt_par    — 16-way parallel MPT using alloy-trie HashBuilder:
//!                  1. Storage roots: rayon par_iter over N changed accounts (independent)
//!                  2. Account trie:  split by first nibble of keccak256(addr) →
//!                     16 independent subtries computed in parallel via rayon,
//!                     assembled into root branch node via RLP + keccak256

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use alloy_trie::{HashBuilder, Nibbles, EMPTY_ROOT_HASH};
use bench::{to_lthash_changes, AccountEntry};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, MemoryDB, Trie};
use lthash::{apply_parallel, apply_sequential, AccountState, StateChange, WorldHash};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{collections::{BTreeMap, HashMap}, sync::Arc};

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

// ── Parallel MPT helpers ──────────────────────────────────────────────────────

/// Compute storage root from a sorted BTreeMap<hashed_slot, value> using HashBuilder.
fn compute_storage_root_hb(slots: &BTreeMap<B256, U256>) -> B256 {
    if slots.is_empty() {
        return EMPTY_ROOT_HASH;
    }
    let mut hb = HashBuilder::default();
    for (hashed_slot, val) in slots {
        hb.add_leaf(Nibbles::unpack(hashed_slot.as_slice()), &rlp_u256(*val));
    }
    hb.root()
}

/// Compute the root of one first-nibble subtrie using HashBuilder.
/// Paths have first nibble stripped (caller has already bucketed by it).
/// Returns None if the bucket is empty (child is absent in branch node).
fn subtrie_root(bucket: &BTreeMap<B256, Vec<u8>>) -> Option<B256> {
    if bucket.is_empty() {
        return None;
    }
    let mut hb = HashBuilder::default();
    for (hashed_addr, encoded) in bucket {
        let full = Nibbles::unpack(hashed_addr.as_slice());
        hb.add_leaf(full.slice(1..), encoded);
    }
    Some(hb.root())
}

/// Assemble an MPT root branch node from 16 optional child hashes.
/// Encodes as RLP list of 17 items (16 children + empty value) then keccak256.
fn assemble_branch_root(children: &[Option<B256>; 16]) -> B256 {
    // Payload: each absent child = 1B (0x80), present = 33B (0xa0 + 32B); + 1B value slot
    let payload: usize =
        children.iter().map(|c| if c.is_some() { 33 } else { 1 }).sum::<usize>() + 1;
    let mut buf = Vec::with_capacity(payload + 4);
    alloy_rlp::Header { list: true, payload_length: payload }.encode(&mut buf);
    for child in children {
        match child {
            None => buf.push(0x80),
            Some(h) => {
                buf.push(0xa0); // 0x80 + 32
                buf.extend_from_slice(h.as_slice());
            }
        }
    }
    buf.push(0x80); // empty value slot
    keccak256(&buf)
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

// ── Parallel MPT state ────────────────────────────────────────────────────────

/// Parallel MPT: accounts bucketed by first nibble of keccak256(addr).
/// - 16 independent subtrie root computations run via rayon
/// - Storage roots also computed in parallel across changed accounts
struct ParMptState {
    /// 16 buckets keyed by first nibble of hashed_addr.
    /// Each bucket: hashed_addr → RLP([nonce, balance, storage_root, code_hash])
    buckets: [BTreeMap<B256, Vec<u8>>; 16],
    /// Per-account storage: hashed_slot → value (BTreeMap for sorted HashBuilder feed)
    storage: HashMap<Address, BTreeMap<B256, U256>>,
}

impl ParMptState {
    fn from_base(accounts: &[AccountEntry]) -> Self {
        let mut buckets: [BTreeMap<B256, Vec<u8>>; 16] = Default::default();
        let mut storage = HashMap::with_capacity(accounts.len());

        for acc in accounts {
            let mut slots: BTreeMap<B256, U256> = BTreeMap::new();
            for &(slot, val) in &acc.storage {
                if !val.is_zero() {
                    slots.insert(keccak256(slot), val);
                }
            }
            let sr = compute_storage_root_hb(&slots);
            let hashed_addr = keccak256(acc.addr);
            let encoded = rlp_account(acc.nonce, acc.balance, sr, acc.code_hash);
            let nibble = (hashed_addr[0] >> 4) as usize;
            buckets[nibble].insert(hashed_addr, encoded);
            storage.insert(acc.addr, slots);
        }

        Self { buckets, storage }
    }

    fn apply_block(&mut self, changes: &[BlockChange]) -> B256 {
        // Step 1 (sequential): read current storage and apply slot changes.
        // Must be sequential because it borrows self.storage per account.
        let prep: Vec<(usize, B256, u64, U256, B256, BTreeMap<B256, U256>)> = changes
            .iter()
            .map(|c| {
                let hashed_addr = keccak256(c.addr);
                let nibble = (hashed_addr[0] >> 4) as usize;
                let mut slots = self.storage.get(&c.addr).cloned().unwrap_or_default();
                for &(slot, _old, new_val) in &c.storage_changes {
                    let hs = keccak256(slot);
                    if new_val.is_zero() {
                        slots.remove(&hs);
                    } else {
                        slots.insert(hs, new_val);
                    }
                }
                (nibble, hashed_addr, c.new_nonce, c.new_balance, c.code_hash, slots)
            })
            .collect();

        // Step 2 (parallel): compute storage roots + encode accounts.
        let encoded: Vec<(usize, B256, Vec<u8>, BTreeMap<B256, U256>)> = prep
            .into_par_iter()
            .map(|(nibble, hashed_addr, nonce, balance, code_hash, slots)| {
                let sr = compute_storage_root_hb(&slots);
                let enc = rlp_account(nonce, balance, sr, code_hash);
                (nibble, hashed_addr, enc, slots)
            })
            .collect();

        // Step 3 (sequential): update buckets and storage state.
        for (change, (nibble, hashed_addr, enc, slots)) in changes.iter().zip(encoded) {
            self.storage.insert(change.addr, slots);
            self.buckets[nibble].insert(hashed_addr, enc);
        }

        // Step 4 (parallel): compute 16 independent subtrie roots.
        let children: [Option<B256>; 16] = self
            .buckets
            .par_iter()
            .map(subtrie_root)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        assemble_branch_root(&children)
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

        // O(n_acc + n_slots) BLAKE3 XOF operations, parallel via rayon.
        // No state-size dependency — cost depends only on changed entries.
        let lthash_changes: Vec<StateChange> =
            block.iter().flat_map(|c| c.to_lthash_changes()).collect();

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
        // Build the trie ONCE and keep it alive across all iterations.
        // Each iter applies the same block delta on top of the previous state —
        // same number of nodes touched per block, same depth, representative timing.
        {
            let mut mpt = MptState::from_base(&base.accounts[..n_acc]);
            let changes = block.clone();
            group.bench_function(BenchmarkId::new("mpt", label), move |b| {
                b.iter(|| mpt.apply_block(&changes));
            });
        }

        // ── mpt_par ──────────────────────────────────────────────────────────
        // Same: one ParMptState kept in memory, block applied each iteration.
        {
            let mut state = ParMptState::from_base(&base.accounts[..n_acc]);
            let changes = block.clone();
            group.bench_function(BenchmarkId::new("mpt_par", label), move |b| {
                b.iter(|| state.apply_block(&changes));
            });
        }
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
