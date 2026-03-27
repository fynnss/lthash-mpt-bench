# LtHash StateDB

**LtHash** (Lattice Hash) as a drop-in replacement for MPT state commitment on high-performance EVM chains. Flat KV + BLAKE3 XOF, no trie overhead.

> As deployed on Solana mainnet ([SIMD-0215](https://github.com/solana-foundation/solana-improvement-documents/pull/215)).

## How it works

```
entry_hash = BLAKE3_XOF(key || value)       →  2048 bytes
world_hash = Σ entry_hash_i                  (wrapping u16 add, commutative)
state_root = BLAKE3(world_hash)              →  32 bytes

// O(changed) incremental update:
world_hash += hash(new_kv) - hash(old_kv)
```

Commutativity means parallel EVM threads compute deltas independently — no synchronization at commit. MPT path rehashing has structural dependencies that force serialization.

Security: SIS lattice problem, 128-bit quantum-safe.

## Structure

```
crates/
├── lthash/     # Core algorithm, no DB dependency
├── state-db/   # RocksDB flat KV + LtHash commitment
└── bench/      # Criterion benchmarks vs MPT
```

## Benchmarks

Base state: **1M accounts**. Each block updates N accounts. **All latency figures are per-block.**

### 1 — Pure hash (in-memory)

`lthash_par`: parallel BLAKE3 XOF deltas, O(N).
`mpt_par`: 16 independent EthTrie subtries (bucketed by first nibble of keccak256), updated in parallel via rayon, O(N × depth / 16).

```bash
cargo bench --bench state_root
```

| Block size | lthash_par | TPS | mpt_par | TPS | Speedup |
|---|---|---|---|---|---|
| 1k   | **0.95 ms** | **1.05M/s** | 5.1 ms  | 197k/s | **5.3×** |
| 10k  | **8.5 ms**  | **1.18M/s** | 37.7 ms | 265k/s | **4.4×** |
| 100k | **82.9 ms** | **1.21M/s** | 248 ms  | 403k/s | **3.0×** |

LtHash TPS is flat across all block sizes (~1.4M/s) — pure O(N), unaffected by total state size.

### 2 — Full commit with RocksDB

`lthash_rdb`: parallel BLAKE3 delta + single atomic WriteBatch (N flat puts).
`mpt_par_rdb`: 16-way parallel `EthTrie<RocksDB>` subtries — same nibble bucketing as `mpt_par`, each subtrie writes its own batch to a shared RocksDB env.

Phase breakdown is printed to stdout on every run (hash vs commit split, TPS per phase).

```bash
cargo bench --bench rw_block
```

Total per-block latency and TPS (Criterion p50):

| Block size | lthash_rdb | TPS | mpt_par_rdb | TPS | Speedup |
|---|---|---|---|---|---|
| 1k   | **2.3 ms**  | **429k/s** | 12.1 ms | 82k/s  | **5.2×** |
| 10k  | **21.8 ms** | **459k/s** | 81.7 ms | 122k/s | **3.7×** |
| 100k | **221 ms**  | **451k/s** | 518 ms  | 193k/s | **2.3×** |

Phase breakdown — printed to stdout on every run (avg over 5 blocks):

- **lthash**: `hash` (BLAKE3 XOF) + `commit` (encode + WriteBatch + `db.write`)
- **mpt**: `insert` (in-memory trie node updates) + `root` (keccak path recompute, writes buffered) + `commit` (`db.write` flush)

| | lt:hash | lt:commit | lt:total | lt:TPS | mpt:insert | mpt:root | mpt:commit | mpt:total | mpt:TPS |
|---|---|---|---|---|---|---|---|---|---|
| 1k   | 1.4 ms  | 0.6 ms   | **2.0 ms**  | **503k/s** | 7.0 ms | 2.7 ms  | 6.1 ms  | 15.8 ms | 63k/s  |
| 10k  | 9.1 ms  | 6.0 ms   | **15.1 ms** | **663k/s** | 41.9 ms | 13.8 ms | 42.6 ms | 98.3 ms | 102k/s |
| 100k | 78.6 ms | 132 ms   | **211 ms**  | **475k/s** | 245 ms | 86.3 ms | 240 ms  | 571 ms  | 175k/s |

Notable patterns:
- **lthash commit** dominates at 100k (132 ms) — `db.write` with a large flat WriteBatch
- **mpt insert** is the heaviest phase (trie path traversal with per-node HashMap lookups for buffering)
- **mpt TPS improves** with block size (63k → 175k/s) — parallel subtries amortize overhead better at larger blocks
- The parallel MPT closes the gap vs serial MPT (~10× old → ~3-5× now) but lthash still wins on predictability and peak TPS

*Apple M-series, 8 cores, `--release`, `lto = "thin"`.*

## Trade-offs

| | LtHash | MPT |
|---|---|---|
| Update cost | O(changed) | O(changed × log n) |
| Parallel-safe | Yes | No |
| Storage overhead | 2 KB | High (branch nodes) |
| Inclusion proofs | No | Yes |
| Quantum-safe | Yes (SIS) | No |

No native inclusion proofs — `eth_getProof` requires an async proof layer or guardian attestation for bridges.

## References

- [Solana SIMD-0215](https://github.com/solana-foundation/solana-improvement-documents/pull/215)
- [Bellare & Micciancio 2019](https://eprint.iacr.org/2019/227)
- [EIP-6800: Verkle Tree](https://eips.ethereum.org/EIPS/eip-6800)
