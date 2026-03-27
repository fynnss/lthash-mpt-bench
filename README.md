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
| 1k   | **0.85 ms** | **1.18M/s** | 4.4 ms  | 227k/s | **5.2×** |
| 10k  | **7.3 ms**  | **1.37M/s** | 36.5 ms | 274k/s | **5.0×** |
| 100k | **71.7 ms** | **1.39M/s** | 230 ms  | 435k/s | **3.2×** |

LtHash TPS is flat across all block sizes (~1.4M/s) — pure O(N), unaffected by total state size.

### 2 — Full commit with RocksDB

`lthash_rdb`: parallel BLAKE3 delta + single atomic WriteBatch (N flat puts).
`mpt_rdb`: EthTrie<RocksDB>, O(depth) random node reads+writes per account.

```bash
cargo bench --bench rw_block
```

| Block size | lthash_rdb | TPS | mpt_rdb | TPS | Speedup |
|---|---|---|---|---|---|
| 1k   | **2.1 ms**  | **476k/s** | 27.9 ms  | 35.8k/s | **13×** |
| 10k  | **21.9 ms** | **457k/s** | 227 ms   | 44k/s   | **10×** |
| 100k | **193 ms**  | **518k/s** | 1,719 ms | 58.2k/s | **9×**  |

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
