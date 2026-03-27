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

Because addition is commutative, parallel EVM threads can independently compute per-change deltas and reduce them with no locks:

```
// Each thread independently:
delta_i = hash(new_kv_i) - hash(old_kv_i)

// Commit phase — no synchronization barrier:
new_state_hash = old_state_hash + Σ delta_i
```

MPT cannot do this: adjacent accounts share branch nodes, so path updates have structural dependencies.

## Storage Schema

Flat KV, no nested tries:

| Column Family | Key | Value |
|---|---|---|
| `accounts` | `addr (20B)` | `nonce (8B LE) \|\| balance (32B LE) \|\| code_hash (32B)` |
| `storages` | `addr (20B) \|\| slot (32B)` | `value (32B BE)` |
| `bytecodes` | `code_hash (32B)` | bytecode bytes |
| `meta` | `"world_hash"` | world hash (2048B), persisted on every commit |

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
use lthash::{WorldHash, EntryHash, StateChange, AccountState, apply_parallel};

// Build world hash from initial state
let mut world = WorldHash::new();
apply_parallel(&mut world, &initial_changes);

// Per-block incremental update — O(changed entries), no state-size dependency
apply_parallel(&mut world, &block_changes);

// 32-byte state root, compatible with existing block header
let state_root: B256 = world.state_root();
```

### `state-db`

```rust
use state_db::StateDb;

let mut db = StateDb::open("/path/to/db")?;

// Apply block changes: updates WorldHash + writes to RocksDB atomically
db.apply_parallel(&changes)?;

// Read flat KV — O(1) point lookup, no trie traversal
let account = db.get_account(&addr)?;
let value   = db.get_storage(&addr, &slot)?;

// State root is always available in O(1)
let root = db.state_root();
```

## Benchmarks

### Setup

Benchmarks simulate a realistic block commit against a **100k-account base state** (4 storage slots each). State change volumes are estimated from 10k TPS / 1s block time with a mixed workload (30% transfers, 40% ERC-20, 30% DeFi), after hot-wallet deduplication.

MPT competitor uses [`eth_trie`](https://crates.io/crates/eth_trie) with `MemoryDB` — a real in-memory incremental MPT (not a full rebuild). **This is a lower bound for MPT**: it has zero DB IO overhead, while a production MPT (reth/geth with MDBX/LevelDB) would be slower due to O(depth) random reads per changed entry.

### Pure hash computation (no DB IO)

```
cargo bench --bench state_root
```

| Scenario | Changes/block | lthash_seq | lthash_par | MPT (mem, lower bound) |
|---|---|---|---|---|
| conservative | 25k | 121 ms | **18 ms** | 282 ms |
| typical | 50k | 249 ms | **40 ms** | 557 ms |
| heavy (DeFi) | 100k | 493 ms | **69 ms** | 1183 ms |

Throughput: `lthash_par` sustains **~1.3M elem/s** (near-linear scaling). MPT sustains ~87k elem/s due to O(depth) keccak256 per entry.

### With RocksDB read + write

```
cargo bench --bench rw_block
```

Includes: `multi_get` reads of old state + WorldHash update + `WriteBatch` write.

| Scenario | Changes/block | lthash_seq | lthash_par | MPT (mem, lower bound) |
|---|---|---|---|---|
| conservative | 25k | 319 ms | **109 ms** | 175 ms |
| typical | 50k | 636 ms | **214 ms** | 319 ms |
| heavy (DeFi) | 100k | 1277 ms | **426 ms** | 577 ms |

`lthash_seq` is slower than `mpt_mem` here because it includes real RocksDB IO; `lthash_par` is still faster. In production, DB writes are async (WAL + background compaction), making the pure hash computation numbers the relevant comparison.

*Tested on Apple M-series, 8 cores, release build with LTO.*

## Running

```bash
# Tests
cargo test

# All benchmarks
cargo bench

# Specific benchmark
cargo bench --bench state_root
cargo bench --bench rw_block
```

## Trade-offs

| | LtHash | MPT |
|---|---|---|
| State root update | O(changed) | O(changed × log n) |
| Parallel update | Yes (no locks) | No (path dependencies) |
| Storage overhead | ~2 KB (world hash only) | High (intermediate nodes) |
| Inclusion proofs | **No** | Yes |
| Quantum-safe | Yes (SIS lattice) | No (keccak256) |
| Light client support | No (needs async proof layer) | Yes |

The core trade-off: **no native inclusion proofs**. `eth_getProof` and storage-proof bridges need adaptation. Mitigation options:

- **Receipt/tx tries**: retained as-is (per-block structures, unaffected)
- **Async proof layer**: a prover network builds an SMT/binary tree from state diffs, periodically anchoring a verifiable root
- **Guardian attestation**: N/M multi-sig for cross-chain bridges (Wormhole-style)

## References

- [Solana SIMD-0215: Accounts Lattice Hash](https://github.com/solana-foundation/solana-improvement-documents/pull/215)
- [Bellare & Micciancio: Securing Update Propagation with Homomorphic Hashing (2019)](https://eprint.iacr.org/2019/227)
- [EIP-6800: Verkle Tree State](https://eips.ethereum.org/EIPS/eip-6800)
- [EIP-7864: Ethereum State Using a Unified Binary Tree](https://eips.ethereum.org/EIPS/eip-7864)
