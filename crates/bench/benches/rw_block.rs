//! Benchmark 2 — Full block commit with RocksDB persistence
//!
//! Both sides use RocksDB. Accounts only, no storage. Base state: 1M accounts.
//! All latency figures are **per-block**.
//!
//! lthash_rdb   — parallel BLAKE3 delta + single atomic WriteBatch (N flat puts)
//! mpt_par_rdb  — 16-way parallel EthTrie<RocksDB> subtries (bucketed by nibble),
//!                parallel inserts + root_hash via rayon, 16 writes to shared RocksDB
//!
//! Phase breakdown (hash vs commit) is printed to stdout before Criterion runs.

use alloy_primitives::{keccak256, B256, U256};
use alloy_rlp::Encodable;
use bench::{gen_accounts, gen_block};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, Trie, DB as TrieDB};
use lthash::{AccountState, StateChange};
use rayon::prelude::*;
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use state_db::StateDb;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use tempfile::TempDir;

const BASE: usize = 1_000_000;
/// Number of rounds used for the per-phase breakdown printout.
const PHASE_ROUNDS: usize = 5;

// ── MPT account encoding ──────────────────────────────────────────────────────

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

// ── Assemble MPT branch root from 16 subtrie roots ───────────────────────────

fn assemble_branch_root(children: &[Option<B256>; 16]) -> B256 {
    let payload: usize =
        children.iter().map(|c| if c.is_some() { 33 } else { 1 }).sum::<usize>() + 1;
    let mut buf = Vec::with_capacity(payload + 4);
    alloy_rlp::Header { list: true, payload_length: payload }.encode(&mut buf);
    for child in children {
        match child {
            None => buf.push(0x80),
            Some(h) => {
                buf.push(0xa0);
                buf.extend_from_slice(h.as_slice());
            }
        }
    }
    buf.push(0x80);
    keccak256(&buf)
}

// ── RocksDB backend for EthTrie ───────────────────────────────────────────────

struct PrefixedRocksDb {
    db: Arc<DB>,
    cf_name: &'static str,
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
        Ok(())
    }
}

const CF_MPT: &str = "mpt_nodes";

fn open_mpt_db<P: AsRef<Path>>(path: P) -> Arc<DB> {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    let cfs = vec![
        ColumnFamilyDescriptor::new("default", Options::default()),
        ColumnFamilyDescriptor::new(CF_MPT, Options::default()),
    ];
    Arc::new(DB::open_cf_descriptors(&opts, path, cfs).unwrap())
}

// ── 16-way parallel MPT backed by RocksDB ────────────────────────────────────
//
// 16 independent EthTrie instances, one per first nibble of keccak256(addr).
// Each holds a PrefixedRocksDb with a 1-byte nibble prefix so all 16 share
// one RocksDB env without key collisions.
//
// apply_block phases:
//   Phase 1 "insert"      — par_iter_mut: N trie.insert() calls (in-memory node updates)
//   Phase 2 "root+commit" — par_iter_mut: trie.root_hash() per subtrie
//                           (keccak256 path recompute + insert_batch to RocksDB)

struct ParMptRdbState {
    subtries: Vec<EthTrie<PrefixedRocksDb>>,
}

impl ParMptRdbState {
    fn from_base(base: &[bench::Account], db: Arc<DB>) -> Self {
        let mut subtries: Vec<EthTrie<PrefixedRocksDb>> = (0..16u8)
            .map(|i| {
                EthTrie::new(Arc::new(PrefixedRocksDb {
                    db: Arc::clone(&db),
                    cf_name: CF_MPT,
                    prefix: vec![i],
                }))
            })
            .collect();

        for (idx, a) in base.iter().enumerate() {
            let hashed = keccak256(a.addr);
            let nibble = (hashed[0] >> 4) as usize;
            subtries[nibble]
                .insert(hashed.as_slice(), &rlp_account(a.nonce, a.balance))
                .unwrap();
            if idx % 100_000 == 99_999 {
                println!("  mpt_par_rdb setup: {}/{BASE}", idx + 1);
            }
        }
        // Materialise all 16 roots (writes initial trie nodes to RocksDB)
        for t in &mut subtries {
            t.root_hash().unwrap();
        }
        Self { subtries }
    }

    /// Apply pre-bucketed block changes; returns state root.
    fn apply_block(&mut self, buckets: &[Vec<([u8; 32], Vec<u8>)>; 16]) -> B256 {
        let (_, _, root) = self.apply_block_timed(buckets);
        root
    }

    /// Apply pre-bucketed changes and return `(insert_dur, root_commit_dur, root)`.
    ///
    /// * `insert_dur`      — parallel trie.insert() calls (in-memory only)
    /// * `root_commit_dur` — parallel root_hash() = keccak path recompute + RocksDB writes
    fn apply_block_timed(
        &mut self,
        buckets: &[Vec<([u8; 32], Vec<u8>)>; 16],
    ) -> (Duration, Duration, B256) {
        // Phase 1: parallel inserts (pure in-memory trie node traversal)
        let t0 = Instant::now();
        self.subtries.par_iter_mut().zip(buckets.par_iter()).for_each(|(trie, bucket)| {
            for (k, v) in bucket {
                trie.insert(k, v).unwrap();
            }
        });
        let insert_dur = t0.elapsed();

        // Phase 2: parallel root_hash — rehashes dirty paths + writes nodes to RocksDB
        let t1 = Instant::now();
        let children: [Option<B256>; 16] = self
            .subtries
            .par_iter_mut()
            .map(|trie| Some(trie.root_hash().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let root = assemble_branch_root(&children);
        let commit_dur = t1.elapsed();

        (insert_dur, commit_dur, root)
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

// ── Phase breakdown helpers ───────────────────────────────────────────────────

fn fmt_dur(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.2}ms", ms)
    } else if ms < 100.0 {
        format!("{:.1}ms", ms)
    } else {
        format!("{:.0}ms", ms)
    }
}

fn fmt_tps(n: usize, d: Duration) -> String {
    let per_sec = n as f64 / d.as_secs_f64();
    if per_sec >= 1_000_000.0 {
        format!("{:.2}M/s", per_sec / 1_000_000.0)
    } else if per_sec >= 1_000.0 {
        format!("{:.0}k/s", per_sec / 1_000.0)
    } else {
        format!("{:.0}/s", per_sec)
    }
}

/// Run `PHASE_ROUNDS` blocks for each scenario and print a breakdown table.
fn print_phase_breakdown(
    scenarios: &[(&str, usize)],
    lthash_db: &mut StateDb,
    mpt_state: &mut ParMptRdbState,
    lthash_changes_map: &[Vec<StateChange>],
    mpt_buckets_map: &[[Vec<([u8; 32], Vec<u8>)>; 16]],
) {
    println!(
        "\n╔══ RocksDB Phase Breakdown — avg over {PHASE_ROUNDS} blocks ══════════════════════════════════════════╗"
    );
    println!(
        "║ {:<6}  {:<42}  {:<44} ║",
        "block", "lthash_rdb", "mpt_par_rdb"
    );
    println!(
        "║ {:<6}  {:<12} {:<12} {:<12} {:>3}  {:<12} {:<14} {:<12} {:>3} ║",
        "", "hash", "commit", "total", "TPS", "insert", "root+commit", "total", "TPS"
    );
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");

    for (idx, &(label, n)) in scenarios.iter().enumerate() {
        let lthash_changes = &lthash_changes_map[idx];
        let buckets = &mpt_buckets_map[idx];

        let mut lh_hash = Duration::ZERO;
        let mut lh_commit = Duration::ZERO;
        let mut mpt_insert = Duration::ZERO;
        let mut mpt_root_commit = Duration::ZERO;

        for _ in 0..PHASE_ROUNDS {
            let (h, c) = lthash_db.apply_parallel_timed(lthash_changes).unwrap();
            lh_hash += h;
            lh_commit += c;

            let (i, rc, _) = mpt_state.apply_block_timed(buckets);
            mpt_insert += i;
            mpt_root_commit += rc;
        }

        let r = PHASE_ROUNDS as u32;
        let lh_h = lh_hash / r;
        let lh_c = lh_commit / r;
        let lh_tot = lh_h + lh_c;
        let mpt_i = mpt_insert / r;
        let mpt_rc = mpt_root_commit / r;
        let mpt_tot = mpt_i + mpt_rc;

        println!(
            "║ {:<6}  {:<12} {:<12} {:<12} {:>6}  {:<12} {:<14} {:<12} {:>6} ║",
            label,
            fmt_dur(lh_h),
            fmt_dur(lh_c),
            fmt_dur(lh_tot),
            fmt_tps(n, lh_tot),
            fmt_dur(mpt_i),
            fmt_dur(mpt_rc),
            fmt_dur(mpt_tot),
            fmt_tps(n, mpt_tot),
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n");
}

// ── Benchmark fixture ─────────────────────────────────────────────────────────

struct Fixture {
    _dir: TempDir,
    base: Vec<bench::Account>,
    lthash_db: StateDb,
    mpt_state: ParMptRdbState,
}

fn build_fixture() -> Fixture {
    println!("Generating {BASE} base accounts…");
    let base = gen_accounts(BASE, 1);
    let dir = tempfile::tempdir().unwrap();

    println!("Building lthash_rdb ({BASE} accounts)…");
    let mut lthash_db = StateDb::open(dir.path().join("lthash")).unwrap();
    let init: Vec<StateChange> = base
        .iter()
        .map(|a| StateChange::Account {
            addr: a.addr,
            old: None,
            new: AccountState { nonce: a.nonce, balance: a.balance, code_hash: B256::ZERO },
        })
        .collect();
    lthash_db.apply_parallel(&init).unwrap();

    println!("Building mpt_par_rdb ({BASE} accounts, 16 subtries)…");
    let mpt_raw = open_mpt_db(dir.path().join("mpt"));
    let mpt_state = ParMptRdbState::from_base(&base, mpt_raw);

    println!("Setup done.\n");
    Fixture { _dir: dir, base, lthash_db, mpt_state }
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_rw(c: &mut Criterion) {
    let mut fix = build_fixture();

    let scenarios: &[(&str, usize)] = &[("1k", 1_000), ("10k", 10_000), ("100k", 100_000)];

    // Pre-compute changes + buckets for all scenarios
    let blocks: Vec<_> = scenarios.iter().map(|&(_, n)| gen_block(&fix.base, n)).collect();

    let lthash_changes_map: Vec<Vec<StateChange>> = blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|d| StateChange::Account { addr: d.addr, old: Some(d.old), new: d.new })
                .collect()
        })
        .collect();

    let mpt_buckets_map: Vec<[Vec<([u8; 32], Vec<u8>)>; 16]> = blocks
        .iter()
        .map(|block| {
            let raw: Vec<([u8; 32], Vec<u8>)> = block
                .iter()
                .map(|d| (*keccak256(d.addr).as_ref(), rlp_account(d.new.nonce, d.new.balance)))
                .collect();
            bucket_entries(&raw)
        })
        .collect();

    // ── Phase breakdown table (printed before Criterion measurements) ─────────
    print_phase_breakdown(
        scenarios,
        &mut fix.lthash_db,
        &mut fix.mpt_state,
        &lthash_changes_map,
        &mpt_buckets_map,
    );

    // ── Criterion: total per-block latency ────────────────────────────────────
    let mut group = c.benchmark_group("rw_block");

    for (idx, &(label, n)) in scenarios.iter().enumerate() {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(BenchmarkId::new("lthash_rdb", label), |b| {
            b.iter(|| fix.lthash_db.apply_parallel(&lthash_changes_map[idx]).unwrap());
        });

        group.bench_function(BenchmarkId::new("mpt_par_rdb", label), |b| {
            b.iter(|| fix.mpt_state.apply_block(&mpt_buckets_map[idx]));
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
