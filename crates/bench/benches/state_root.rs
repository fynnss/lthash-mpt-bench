//! Benchmark 1 — Pure hash computation (no DB IO)
//!
//! LtHash (parallel BLAKE3 XOF) vs MPT (16-way parallel subtrie, incremental).
//! Accounts only, no storage. Base state: 1M accounts.
//!
//! Scenarios: one block updates 1k / 10k / 100k accounts.
//!
//! lthash_par — clone 2048-byte accumulator, N parallel BLAKE3 XOF deltas, finalise root
//!              cost: O(N), independent of total state size
//!
//! mpt_par    — 16 independent EthTrie<MemoryDB> subtries (bucketed by first nibble of
//!              keccak256(addr)), changed buckets updated in parallel via rayon,
//!              subtrie roots assembled into a branch-node root with keccak256
//!              cost: O(N × depth / 16) — truly incremental, parallel

use alloy_primitives::{keccak256, B256, U256};
use alloy_rlp::Encodable;
use bench::{gen_accounts, gen_block};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, MemoryDB, Trie};
use lthash::{apply_parallel, apply_sequential, AccountState, StateChange, WorldHash};
use rayon::prelude::*;
use std::sync::Arc;

const BASE: usize = 1_000_000;

// ── MPT account encoding: RLP([nonce, balance, EMPTY_ROOT, ZERO_CODE_HASH]) ──

fn rlp_account(nonce: u64, balance: U256) -> Vec<u8> {
    let storage_root = alloy_trie::EMPTY_ROOT_HASH;
    let code_hash = B256::ZERO;
    let payload =
        nonce.length() + balance.length() + storage_root.length() + code_hash.length();
    let mut buf = Vec::with_capacity(payload + 4);
    alloy_rlp::Header { list: true, payload_length: payload }.encode(&mut buf);
    nonce.encode(&mut buf);
    balance.encode(&mut buf);
    storage_root.encode(&mut buf);
    code_hash.encode(&mut buf);
    buf
}

// ── Assemble MPT branch root from 16 child hashes ────────────────────────────
//
// RLP-encode a 17-item branch node (16 child slots + empty value slot),
// then keccak256 the result. Standard Ethereum MPT encoding.

fn assemble_branch_root(children: &[Option<B256>; 16]) -> B256 {
    // Each absent child = 1B (0x80), present = 33B (0xa0 + 32B); + 1B value slot
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

// ── 16-way parallel MPT ───────────────────────────────────────────────────────

/// 16 independent EthTrie<MemoryDB> subtries, one per first nibble of keccak256(addr).
///
/// Each subtrie is fully independent — no shared state, no locks. Rayon can
/// update all 16 in parallel during a block commit.
struct ParMptState {
    subtries: Vec<EthTrie<MemoryDB>>,
}

impl ParMptState {
    fn from_base(accounts: &[bench::Account]) -> Self {
        let mut subtries: Vec<EthTrie<MemoryDB>> = (0..16)
            .map(|_| EthTrie::new(Arc::new(MemoryDB::new(true))))
            .collect();

        for a in accounts {
            let hashed = keccak256(a.addr);
            let nibble = (hashed[0] >> 4) as usize;
            subtries[nibble].insert(hashed.as_slice(), &rlp_account(a.nonce, a.balance)).unwrap();
        }
        // Materialise all 16 roots
        for t in &mut subtries {
            t.root_hash().unwrap();
        }
        Self { subtries }
    }

    /// Apply pre-bucketed changes in parallel and return the new state root.
    ///
    /// `buckets[i]` contains (hashed_addr, encoded_account) pairs whose
    /// keccak256(addr) starts with nibble `i`.
    fn apply_block(&mut self, buckets: &[Vec<([u8; 32], Vec<u8>)>; 16]) -> B256 {
        let children: [Option<B256>; 16] = self
            .subtries
            .par_iter_mut()
            .zip(buckets.par_iter())
            .map(|(trie, bucket)| {
                if bucket.is_empty() {
                    // No changes in this bucket — root unchanged; still need the value.
                    Some(trie.root_hash().unwrap())
                } else {
                    for (hashed_addr, encoded) in bucket {
                        trie.insert(hashed_addr, encoded).unwrap();
                    }
                    Some(trie.root_hash().unwrap())
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        assemble_branch_root(&children)
    }
}

// ── Pre-bucket block changes by first nibble ──────────────────────────────────

fn bucket_entries(entries: &[([u8; 32], Vec<u8>)]) -> [Vec<([u8; 32], Vec<u8>)>; 16] {
    let mut buckets: [Vec<([u8; 32], Vec<u8>)>; 16] = Default::default();
    for (hashed_addr, encoded) in entries {
        let nibble = (hashed_addr[0] >> 4) as usize;
        buckets[nibble].push((*hashed_addr, encoded.clone()));
    }
    buckets
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_state_root(c: &mut Criterion) {
    // ── One-time setup ────────────────────────────────────────────────────────

    println!("Generating {BASE} base accounts…");
    let base = gen_accounts(BASE, 1);

    println!("Building LtHash world hash from {BASE} accounts…");
    let init_changes: Vec<StateChange> = base
        .iter()
        .map(|a| StateChange::Account {
            addr: a.addr,
            old: None,
            new: AccountState { nonce: a.nonce, balance: a.balance, code_hash: B256::ZERO },
        })
        .collect();
    let mut base_world = WorldHash::new();
    apply_sequential(&mut base_world, &init_changes);

    println!("Building ParMptState (16 × EthTrie<MemoryDB>) from {BASE} accounts…");
    let mut par_mpt = ParMptState::from_base(&base);

    println!("Setup done. Starting benchmarks…\n");

    // ── Scenarios ─────────────────────────────────────────────────────────────

    let scenarios: &[(&str, usize)] = &[("1k", 1_000), ("10k", 10_000), ("100k", 100_000)];
    let mut group = c.benchmark_group("state_root");

    for &(label, n) in scenarios {
        let block = gen_block(&base, n);
        group.throughput(Throughput::Elements(n as u64));

        // Pre-compute lthash StateChanges
        let lthash_changes: Vec<StateChange> = block
            .iter()
            .map(|d| StateChange::Account { addr: d.addr, old: Some(d.old), new: d.new })
            .collect();

        // Pre-compute hashed+encoded pairs and bucket them for mpt_par
        let raw_entries: Vec<([u8; 32], Vec<u8>)> = block
            .iter()
            .map(|d| (*keccak256(d.addr).as_ref(), rlp_account(d.new.nonce, d.new.balance)))
            .collect();
        let buckets = bucket_entries(&raw_entries);

        // ── lthash_par ───────────────────────────────────────────────────────
        // Clone 2048-byte accumulator, apply N BLAKE3 XOF deltas in parallel,
        // finalise 32-byte root. O(N), independent of total state size.
        group.bench_with_input(
            BenchmarkId::new("lthash_par", label),
            &lthash_changes,
            |b, changes| {
                b.iter_batched(
                    || base_world.clone(), // 2048 bytes
                    |mut world| {
                        apply_parallel(&mut world, changes);
                        world.state_root()
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // ── mpt_par ──────────────────────────────────────────────────────────
        // 16 independent EthTrie subtries updated in parallel via rayon.
        // Each subtrie covers one first-nibble bucket (~62.5k accounts each).
        // Changed subtries are updated and re-rooted; all 16 roots assembled
        // into a branch-node root. O(N × depth / 16), truly incremental.
        group.bench_function(BenchmarkId::new("mpt_par", label), |b| {
            b.iter(|| par_mpt.apply_block(&buckets));
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(8));
    targets = bench_state_root
}
criterion_main!(benches);
