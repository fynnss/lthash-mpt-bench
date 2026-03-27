# LtHash StateDB

A Rust implementation of **Lattice Hash (LtHash)** as a state commitment scheme for high-performance EVM chains, with a RocksDB-backed flat KV state database and benchmarks comparing it against the Merkle Patricia Trie (MPT).

## Background

At 10k+ TPS, MPT becomes the dominant bottleneck in EVM block production:

- Every state change requires O(log n) random IO to update trie paths
- Intermediate trie nodes dominate storage (Ethereum state DB > 1 TB, mostly non-leaf overhead)
- The serial commit path prevents parallel EVM gains — no matter how fast execution is, trie updates still serialize at block end

This project implements and benchmarks an alternative: **Flat KV + LtHash**, as deployed on Solana mainnet ([SIMD-0215](https://github.com/solana-foundation/solana-improvement-documents/pull/215)).

## How LtHash Works

```
// Per entry
entry_hash = BLAKE3_XOF("lthash-evm-state-v1" || key || value)  →  2048 bytes (1024 × u16)

// Aggregate (commutative, associative)
world_hash = Σ entry_hash_i   (wrapping u16 addition)

// O(1) incremental update
world_hash += hash(new_kv) - hash(old_kv)

// State root
state_root = BLAKE3(world_hash)  →  32 bytes
```

Security is based on the **SIS lattice problem** — 128-bit quantum-safe, same family as NIST PQ standards (ML-KEM/ML-DSA). No trusted setup.

### Why it enables parallelism

Because wrapping u16 addition is commutative and associative, parallel EVM threads can independently compute per-change deltas and reduce them with no locks:

```
// Each thread independently (Block-STM / OCC workers):
delta_i = hash(new_kv_i) - hash(old_kv_i)

// Commit phase — single reduce, no synchronization barrier:
new_state_hash = old_state_hash + Σ delta_i
```

MPT cannot do this: adjacent accounts share branch nodes, so path rehashing has structural dependencies that force serialization.

## Storage Schema

Flat KV, no nested tries:

| Column Family | Key | Value |
|---|---|---|
| `accounts`  | `addr (20B)` | `nonce (8B LE) \|\| balance (32B LE) \|\| code_hash (32B)` |
| `storages`  | `addr (20B) \|\| slot (32B)` | `value (32B BE)` |
| `bytecodes` | `code_hash (32B)` | bytecode bytes |
| `meta`      | `"world_hash"` | world hash (2048B), persisted on every commit |

Compared to Reth V2, this removes 4 tables: `HashedAccounts`, `HashedStorages`, `AccountsTrie`, `StoragesTrie`.

## Project Structure

```
crates/
├── lthash/     # Core LtHash algorithm — pure, no DB dependency
├── state-db/   # RocksDB-backed flat KV state + LtHash commitment
└── bench/      # Criterion benchmarks vs MPT
```

### `lthash`

```rust
use lthash::{WorldHash, StateChange, AccountState, apply_parallel};

// Build world hash from initial state (e.g. genesis or snapshot load)
let mut world = WorldHash::new();
apply_parallel(&mut world, &initial_changes);

// Per-block incremental update — O(changed entries), independent of total state size
apply_parallel(&mut world, &block_changes);

// 32-byte state root, compatible with existing block header format
let state_root: B256 = world.state_root();
```

### `state-db`

```rust
use state_db::StateDb;

let mut db = StateDb::open("/path/to/db")?;

// Apply block changes: parallel WorldHash update + atomic RocksDB WriteBatch
db.apply_parallel(&changes)?;

// Flat reads — O(1) point lookup, no trie traversal
let account = db.get_account(&addr)?;
let value   = db.get_storage(&addr, &slot)?;

// State root always available in O(1), no recomputation needed
let root = db.state_root();
```

## Benchmarks

### Workload model

Benchmarks use a **100k-account base state** (4 storage slots per account) representing an established chain. Block-level state change volumes are derived from a 10k TPS / 1s block time assumption with a realistic mixed workload:

| Tx type | Share | Acct writes/tx | Storage writes/tx |
|---|---|---|---|
| ETH transfer | 30% | 2 (sender + receiver) | 0 |
| ERC-20 transfer | 40% | 3 (sender, receiver, contract) | 2 (balances) |
| DeFi (swap/lend) | 30% | ~7 | ~10 |

After applying ~40% deduplication for hot wallets/contracts, three scenarios are tested:

| Scenario | Unique accounts | Unique storage slots | Total changes | Representative workload |
|---|---|---|---|---|
| `conservative_25k` | 11,700 | 13,300 | **25,000** | Token transfers dominate, high address reuse |
| `typical_50k` | 23,400 | 26,600 | **50,000** | Balanced DeFi + transfer mix |
| `heavy_100k` | 46,800 | 53,200 | **100,000** | DeFi-heavy, low address reuse |

### Benchmark 1 — Pure hash computation (no DB IO)

Models the **async-write path**: execution results are buffered in memory, DB writes happen asynchronously via WAL. Only the state root computation is on the critical path.

```
cargo bench --bench state_root
```

Three competitors operate entirely in memory:
- **`lthash_par`**: rayon parallel BLAKE3 XOF per entry + wrapping-add reduce
- **`mpt`**: [`eth_trie`](https://crates.io/crates/eth_trie) with `MemoryDB` — real incremental in-memory MPT (not a full rebuild); represents the **lower bound** for MPT since there is zero DB IO
- **`mpt_par`**: 16-way parallel MPT using alloy-trie `HashBuilder`:
  1. Storage roots: rayon `par_iter` over N changed accounts (each independent)
  2. Account trie: split by first nibble of `keccak256(addr)` → 16 independent subtries computed in parallel, assembled into root branch node via RLP + keccak256

| Scenario | lthash_par | mpt_par | MPT (serial) | vs mpt_par | vs mpt |
|---|---|---|---|---|---|
| conservative 25k | **19 ms** | 54 ms | 271 ms | 2.8× | 14× |
| typical 50k | **36 ms** | 115 ms | 576 ms | 3.2× | 16× |
| heavy 100k | **77 ms** | 230 ms | 1,126 ms | 3.0× | 15× |

Assuming 1s block time and the given scenario represents ~10k TPS:

| Scenario | lthash_par | mpt_par | MPT (serial) |
|---|---|---|---|
| conservative 25k | **515k TPS** | 185k TPS | 37k TPS |
| typical 50k | **275k TPS** | 87k TPS | 17k TPS |
| heavy 100k | **130k TPS** | 43k TPS | 8.9k TPS ❌ |

> Max TPS capacity if state-root computation is the sole bottleneck. MPT (serial) cannot sustain 10k TPS under a heavy DeFi workload with 1s blocks.

`lthash_par` sustains **~1.3M state changes/s**. `mpt_par` reaches ~435k/s by parallelising storage root computation and the 16 independent subtries. Serial MPT is capped at ~87k/s due to O(depth) keccak256 operations per changed entry.

### Benchmark 2 — Full pipeline with RocksDB (both sides)

Models the **sync-write path**: read old state → compute new root → persist to RocksDB, all on the critical path. Both competitors use the same RocksDB engine for a fair comparison.

```
cargo bench --bench rw_block
```

- **`lthash_par`**: `multi_get` (1 read/entry, flat KV) + parallel BLAKE3 + `WriteBatch` (1 write/entry)
- **`mpt_rocksdb`**: `eth_trie` with a custom `rocksdb` backend — trie node reads are O(depth) per changed entry, node writes are O(depth) per changed entry

| Scenario | lthash_par | mpt_rocksdb | Speedup |
|---|---|---|---|
| conservative 25k | **136 ms** | 379 ms | 2.8× |
| typical 50k | **236 ms** | 734 ms | 3.1× |
| heavy 100k | **460 ms** | 1,418 ms | 3.1× |

The gap narrows vs. benchmark 1 because both sides now pay RocksDB IO. LtHash's advantage comes from flat access patterns (1 read and 1 write per entry) vs. MPT's O(depth) random node accesses per entry.

*All results: Apple M-series, 8 cores, `--release` with `lto = "thin"`.*

### Running the benchmarks

```bash
cargo test                       # unit tests
cargo bench                      # all benchmarks
cargo bench --bench state_root   # pure hash only
cargo bench --bench rw_block     # full read-write pipeline
```

## Trade-offs

| | LtHash | MPT |
|---|---|---|
| State root update | O(changed) | O(changed × log n) |
| Parallel update | Yes — commutative, no locks | No — path dependencies |
| Storage overhead | ~2 KB (world hash only) | High (branch/extension nodes) |
| Inclusion proofs | **No** | Yes |
| Quantum-safe | Yes (SIS lattice) | No (keccak256) |
| Light client support | No — needs async proof layer | Yes |

The core trade-off is **no native inclusion proofs**. `eth_getProof` and storage-proof bridges require adaptation. Mitigation options:

- **Receipt/tx tries**: retained as-is — per-block structures unaffected by this change
- **Async proof layer**: a prover network builds an SMT/binary tree from state diffs and periodically anchors a verifiable root back on-chain
- **Guardian attestation**: N/M multi-sig for cross-chain bridges (Wormhole-style), available from day one

## References

- [Solana SIMD-0215: Accounts Lattice Hash](https://github.com/solana-foundation/solana-improvement-documents/pull/215)
- [Bellare & Micciancio: Securing Update Propagation with Homomorphic Hashing (2019)](https://eprint.iacr.org/2019/227)
- [EIP-6800: Verkle Tree State](https://eips.ethereum.org/EIPS/eip-6800)
- [EIP-7864: Ethereum State Using a Unified Binary Tree](https://eips.ethereum.org/EIPS/eip-7864)
