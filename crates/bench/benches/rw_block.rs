//! Benchmark 2 — Full block commit with RocksDB persistence
//!
//! Both sides use RocksDB. Accounts only, no storage. Base state: 1M accounts.
//! All latency figures are **per-block**.
//!
//! Three-phase breakdown (printed to stdout before Criterion runs):
//!
//! lthash_rdb:
//!   hash  — parallel BLAKE3 XOF delta computation
//!   build — encode accounts + assemble WriteBatch in memory
//!   write — db.write(batch): RocksDB WAL / memtable
//!
//! mpt_par_rdb (16-way parallel EthTrie):
//!   insert — par trie.insert() calls: in-memory node traversal
//!   root   — par root_hash(): keccak path recompute, dirty nodes buffered (no DB write yet)
//!   write  — flush all 16 pending WriteBatches to RocksDB

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
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tempfile::TempDir;

const BASE: usize = 1_000_000;
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

// ── RocksDB backend for EthTrie — with write buffering ───────────────────────
//
// Writes (insert / insert_batch / remove) are buffered in a HashMap instead of
// going to RocksDB immediately.  Reads (get) check the buffer first, then fall
// back to RocksDB — this is required because eth_trie verifies the root node
// exists via get() immediately after root_hash() commits dirty nodes.
//
// Call flush_pending() to convert the buffer to a WriteBatch and write to DB.
// This lets us time the root-hash phase (keccak recompute + buffer fill)
// independently from the actual RocksDB write phase.

use std::collections::HashMap;

struct PrefixedRocksDb {
    db: Arc<DB>,
    cf_name: &'static str,
    prefix: Vec<u8>,
    /// Pending writes: prefixed_key → Some(value) for puts, None for deletes.
    pending: Mutex<HashMap<Vec<u8>, Option<Vec<u8>>>>,
}

impl PrefixedRocksDb {
    fn new(db: Arc<DB>, cf_name: &'static str, prefix: Vec<u8>) -> Arc<Self> {
        Arc::new(Self { db, cf_name, prefix, pending: Mutex::new(HashMap::new()) })
    }

    fn prefixed(&self, key: &[u8]) -> Vec<u8> {
        let mut k = Vec::with_capacity(self.prefix.len() + key.len());
        k.extend_from_slice(&self.prefix);
        k.extend_from_slice(key);
        k
    }

    fn cf(&self) -> &rocksdb::ColumnFamily {
        self.db.cf_handle(self.cf_name).unwrap()
    }

    /// Flush buffered writes to RocksDB via a single WriteBatch, then clear buffer.
    fn flush_pending(&self) -> Result<(), rocksdb::Error> {
        let mut pending = self.pending.lock().unwrap();
        if pending.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for (k, v) in pending.drain() {
            match v {
                Some(val) => batch.put_cf(self.cf(), &k, val),
                None => batch.delete_cf(self.cf(), &k),
            }
        }
        self.db.write(batch)
    }
}

impl TrieDB for PrefixedRocksDb {
    type Error = rocksdb::Error;

    /// Read-through: check pending buffer first, then RocksDB.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        let pk = self.prefixed(key);
        if let Some(entry) = self.pending.lock().unwrap().get(&pk) {
            return Ok(entry.clone());
        }
        self.db.get_cf(self.cf(), &pk)
    }
    fn insert(&self, key: &[u8], value: Vec<u8>) -> Result<(), Self::Error> {
        self.pending.lock().unwrap().insert(self.prefixed(key), Some(value));
        Ok(())
    }
    fn insert_batch(&self, keys: Vec<Vec<u8>>, values: Vec<Vec<u8>>) -> Result<(), Self::Error> {
        let mut pending = self.pending.lock().unwrap();
        for (k, v) in keys.into_iter().zip(values) {
            pending.insert(self.prefixed(&k), Some(v));
        }
        Ok(())
    }
    fn remove(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.pending.lock().unwrap().insert(self.prefixed(key), None);
        Ok(())
    }
    fn remove_batch(&self, keys: &[Vec<u8>]) -> Result<(), Self::Error> {
        let mut pending = self.pending.lock().unwrap();
        for k in keys {
            pending.insert(self.prefixed(k), None);
        }
        Ok(())
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

struct ParMptRdbState {
    subtries: Vec<EthTrie<PrefixedRocksDb>>,
    backends: Vec<Arc<PrefixedRocksDb>>,
}

impl ParMptRdbState {
    fn from_base(base: &[bench::Account], db: Arc<DB>) -> Self {
        let mut backends = Vec::with_capacity(16);
        let mut subtries = Vec::with_capacity(16);

        for i in 0u8..16 {
            let backend = PrefixedRocksDb::new(Arc::clone(&db), CF_MPT, vec![i]);
            subtries.push(EthTrie::new(Arc::clone(&backend)));
            backends.push(backend);
        }

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

        // Materialise roots and flush initial trie nodes to RocksDB
        for (trie, backend) in subtries.iter_mut().zip(backends.iter()) {
            trie.root_hash().unwrap();
            backend.flush_pending().unwrap();
        }

        Self { subtries, backends }
    }

    /// Apply block (all 3 phases). Used by Criterion for total latency.
    fn apply_block(&mut self, buckets: &[Vec<([u8; 32], Vec<u8>)>; 16]) -> B256 {
        let (_, _, _, root) = self.apply_block_timed(buckets);
        root
    }

    /// Apply block with per-phase timing.
    /// Returns `(insert_dur, root_dur, write_dur, state_root)`.
    ///
    /// * `insert_dur` — par trie.insert(): in-memory node traversal only
    /// * `root_dur`   — par root_hash(): keccak path recompute, writes buffered
    /// * `write_dur`  — flush 16 pending batches to RocksDB
    fn apply_block_timed(
        &mut self,
        buckets: &[Vec<([u8; 32], Vec<u8>)>; 16],
    ) -> (Duration, Duration, Duration, B256) {
        // Phase 1: parallel trie.insert (pure in-memory)
        let t0 = Instant::now();
        self.subtries.par_iter_mut().zip(buckets.par_iter()).for_each(|(trie, bucket)| {
            for (k, v) in bucket {
                trie.insert(k, v).unwrap();
            }
        });
        let insert_dur = t0.elapsed();

        // Phase 2: parallel root_hash — keccak recompute, dirty nodes buffered
        let t1 = Instant::now();
        let children: [Option<B256>; 16] = self
            .subtries
            .par_iter_mut()
            .map(|trie| Some(trie.root_hash().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let root = assemble_branch_root(&children);
        let root_dur = t1.elapsed();

        // Phase 3: flush all 16 pending batches to RocksDB
        let t2 = Instant::now();
        for backend in &self.backends {
            backend.flush_pending().unwrap();
        }
        let write_dur = t2.elapsed();

        (insert_dur, root_dur, write_dur, root)
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
    } else {
        format!("{:.0}k/s", per_sec / 1_000.0)
    }
}

fn print_phase_breakdown(
    scenarios: &[(&str, usize)],
    lthash_db: &mut StateDb,
    mpt_state: &mut ParMptRdbState,
    lthash_changes_map: &[Vec<StateChange>],
    mpt_buckets_map: &[[Vec<([u8; 32], Vec<u8>)>; 16]],
) {
    println!(
        "\n╔══ RocksDB Phase Breakdown — avg over {PHASE_ROUNDS} blocks ══════════════════════════════════════════════╗"
    );
    println!(
        "║  {:<6}  {:<10} {:<10} {:<10} {:<7}    {:<10} {:<10} {:<10} {:<10} {:<7}  ║",
        "block",
        "lt:hash", "lt:commit", "lt:total", "lt:TPS",
        "mpt:insert", "mpt:root", "mpt:commit", "mpt:total", "mpt:TPS",
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for (idx, &(label, n)) in scenarios.iter().enumerate() {
        let lthash_changes = &lthash_changes_map[idx];
        let buckets = &mpt_buckets_map[idx];

        let mut lh_hash = Duration::ZERO;
        let mut lh_commit = Duration::ZERO;
        let mut mpt_insert = Duration::ZERO;
        let mut mpt_root = Duration::ZERO;
        let mut mpt_commit = Duration::ZERO;

        for _ in 0..PHASE_ROUNDS {
            let (h, c) = lthash_db.apply_parallel_timed(lthash_changes).unwrap();
            lh_hash += h;
            lh_commit += c;

            let (i, r, w, _) = mpt_state.apply_block_timed(buckets);
            mpt_insert += i;
            mpt_root += r;
            mpt_commit += w;
        }

        let r = PHASE_ROUNDS as u32;
        let lh_h = lh_hash / r;
        let lh_c = lh_commit / r;
        let lh_tot = lh_h + lh_c;
        let mpt_i = mpt_insert / r;
        let mpt_r = mpt_root / r;
        let mpt_c = mpt_commit / r;
        let mpt_tot = mpt_i + mpt_r + mpt_c;

        println!(
            "║  {:<6}  {:<10} {:<10} {:<10} {:<7}    {:<10} {:<10} {:<10} {:<10} {:<7}  ║",
            label,
            fmt_dur(lh_h), fmt_dur(lh_c), fmt_dur(lh_tot), fmt_tps(n, lh_tot),
            fmt_dur(mpt_i), fmt_dur(mpt_r), fmt_dur(mpt_c), fmt_dur(mpt_tot), fmt_tps(n, mpt_tot),
        );
    }

    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    );
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

    // ── Phase breakdown (printed before Criterion) ────────────────────────────
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
