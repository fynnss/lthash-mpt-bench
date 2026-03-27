//! Benchmark 2 — Full block commit with RocksDB persistence
//!
//! Both sides use RocksDB — apples-to-apples storage comparison.
//! Accounts only, no storage. Base state: 1M accounts.
//!
//! Scenarios: one block updates 1k / 10k / 100k accounts.
//!
//! lthash_rdb — parallel BLAKE3 delta + atomic flat-KV WriteBatch
//!              reads: O(N) flat (pre-state cache; 1 get/account)
//!              writes: O(N) flat puts + 1 world-hash put
//!
//! mpt_rdb    — EthTrie<RocksDB> incremental update + root_hash
//!              reads: O(N × depth) trie-node gets per insert
//!              writes: O(N × depth) trie-node puts per insert

use alloy_primitives::{keccak256, B256, U256};
use alloy_rlp::Encodable;
use bench::{gen_accounts, gen_block};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use eth_trie::{EthTrie, Trie, DB as TrieDB};
use lthash::{AccountState, StateChange};
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use state_db::StateDb;
use std::{path::Path, sync::Arc};
use tempfile::TempDir;

const BASE: usize = 1_000_000;

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

// ── Benchmark fixture ─────────────────────────────────────────────────────────

struct Fixture {
    _dir: TempDir,
    base: Vec<bench::Account>,
    lthash_db: StateDb,
    mpt_trie: EthTrie<PrefixedRocksDb>,
}

fn build_fixture() -> Fixture {
    println!("Generating {BASE} base accounts…");
    let base = gen_accounts(BASE, 1);
    let dir = tempfile::tempdir().unwrap();

    // LtHash flat KV — fast bulk insert via WriteBatch
    println!("Building LtHash flat KV ({BASE} accounts)…");
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

    // MPT RocksDB — insert account trie nodes
    println!("Building MPT RocksDB ({BASE} accounts)…");
    let mpt_raw = open_mpt_db(dir.path().join("mpt"));
    let backend = Arc::new(PrefixedRocksDb {
        db: Arc::clone(&mpt_raw),
        cf_name: CF_MPT,
        prefix: vec![0x00],
    });
    let mut mpt_trie = EthTrie::new(backend);
    for (i, a) in base.iter().enumerate() {
        mpt_trie
            .insert(keccak256(a.addr).as_slice(), &rlp_account(a.nonce, a.balance))
            .unwrap();
        if i % 100_000 == 99_999 {
            println!("  MPT: {}/{BASE}", i + 1);
        }
    }
    mpt_trie.root_hash().unwrap();

    println!("Setup done. Starting benchmarks…\n");
    Fixture { _dir: dir, base, lthash_db, mpt_trie }
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

fn bench_rw(c: &mut Criterion) {
    let mut fix = build_fixture();

    let scenarios: &[(&str, usize)] = &[("1k", 1_000), ("10k", 10_000), ("100k", 100_000)];
    let mut group = c.benchmark_group("rw_block");

    for &(label, n) in scenarios {
        let block = gen_block(&fix.base, n);
        group.throughput(Throughput::Elements(n as u64));

        // ── lthash_rdb ───────────────────────────────────────────────────────
        // Old values come from the EVM pre-state cache (stored in deltas).
        // Pipeline: parallel BLAKE3 delta + atomic WriteBatch (N flat puts).
        let lthash_changes: Vec<StateChange> = block
            .iter()
            .map(|d| StateChange::Account { addr: d.addr, old: Some(d.old), new: d.new })
            .collect();

        group.bench_function(BenchmarkId::new("lthash_rdb", label), |b| {
            b.iter(|| fix.lthash_db.apply_parallel(&lthash_changes).unwrap());
        });

        // ── mpt_rdb ──────────────────────────────────────────────────────────
        // EthTrie<RocksDB>: each insert reads O(depth) nodes from RocksDB,
        // rehashes the path, and writes O(depth) updated nodes back.
        let mpt_entries: Vec<([u8; 32], Vec<u8>)> = block
            .iter()
            .map(|d| (*keccak256(d.addr).as_ref(), rlp_account(d.new.nonce, d.new.balance)))
            .collect();

        group.bench_function(BenchmarkId::new("mpt_rdb", label), |b| {
            b.iter(|| {
                for (hashed_addr, encoded) in &mpt_entries {
                    fix.mpt_trie.insert(hashed_addr, encoded).unwrap();
                }
                fix.mpt_trie.root_hash().unwrap()
            });
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
