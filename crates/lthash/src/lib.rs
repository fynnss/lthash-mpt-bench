//! Lattice Hash (LtHash) state commitment scheme.
//!
//! Based on Solana SIMD-0215 and Bellare & Micciancio (2019).
//!
//! ## How it works
//!
//! 1. **Per entry**: `BLAKE3_XOF(key || value)` → 2048-byte vector (1024 × u16)
//! 2. **Aggregate**: `WorldHash = Σ all entry vectors` (wrapping u16 addition)
//! 3. **Update**: `new = old − hash(prev_value) + hash(new_value)` → O(1)
//! 4. **StateRoot**: `BLAKE3(WorldHash)` → 32 bytes
//!
//! Security is based on the SIS lattice problem — 128-bit quantum-safe.

use alloy_primitives::{Address, B256, U256};
use rayon::prelude::*;

/// Size of the world hash in bytes (1024 × u16).
pub const WORLD_HASH_BYTES: usize = 2048;
/// Number of u16 elements in the world hash.
pub const WORLD_HASH_LEN: usize = WORLD_HASH_BYTES / 2;

/// Domain separation for BLAKE3 XOF derivation.
const LTHASH_DOMAIN: &[u8] = b"lthash-evm-state-v1";

/// A 2048-byte lattice hash vector (1024 × u16, wrapping arithmetic).
///
/// This is the hash of a single KV entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryHash([u16; WORLD_HASH_LEN]);

impl EntryHash {
    /// Compute the lattice hash for an account entry.
    ///
    /// Key schema: `[0x00] || address(20B)`
    /// Value schema: `nonce(8B LE) || balance(32B LE) || code_hash(32B)`
    pub fn for_account(addr: &Address, nonce: u64, balance: U256, code_hash: B256) -> Self {
        let mut key = [0u8; 21];
        key[0] = 0x00;
        key[1..21].copy_from_slice(addr.as_slice());

        let mut value = [0u8; 72];
        value[0..8].copy_from_slice(&nonce.to_le_bytes());
        value[8..40].copy_from_slice(&balance.to_le_bytes::<32>());
        value[40..72].copy_from_slice(code_hash.as_slice());

        Self::compute(&key, &value)
    }

    /// Compute the lattice hash for a storage entry.
    ///
    /// Key schema: `[0x01] || address(20B) || slot(32B)`
    /// Value schema: `value(32B)`
    pub fn for_storage(addr: &Address, slot: B256, value: U256) -> Self {
        let mut key = [0u8; 53];
        key[0] = 0x01;
        key[1..21].copy_from_slice(addr.as_slice());
        key[21..53].copy_from_slice(slot.as_slice());

        let val_bytes = value.to_be_bytes::<32>();
        Self::compute(&key, &val_bytes)
    }

    /// Raw BLAKE3 XOF computation: feeds `key || value` and reads 2048 bytes.
    pub fn compute(key: &[u8], value: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(LTHASH_DOMAIN);
        hasher.update(b"\x00"); // domain separator null byte
        hasher.update(key);
        hasher.update(b"\x00"); // separator between key and value
        hasher.update(value);

        let mut bytes = [0u8; WORLD_HASH_BYTES];
        hasher.finalize_xof().fill(&mut bytes);

        let mut elements = [0u16; WORLD_HASH_LEN];
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            elements[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Self(elements)
    }

    pub fn as_elements(&self) -> &[u16; WORLD_HASH_LEN] {
        &self.0
    }
}

/// The accumulated world hash over all KV entries.
///
/// `WorldHash = Σ EntryHash(kᵢ, vᵢ)` with wrapping u16 addition.
/// `StateRoot = BLAKE3(world_hash_bytes)` → 32 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldHash([u16; WORLD_HASH_LEN]);

impl Default for WorldHash {
    fn default() -> Self {
        Self([0u16; WORLD_HASH_LEN])
    }
}

impl WorldHash {
    /// Create a zero (empty) world hash.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry hash (insert or re-insert a KV entry).
    #[inline]
    pub fn add(&mut self, entry: &EntryHash) {
        for (a, b) in self.0.iter_mut().zip(entry.0.iter()) {
            *a = a.wrapping_add(*b);
        }
    }

    /// Remove an entry hash (delete a KV entry).
    #[inline]
    pub fn remove(&mut self, entry: &EntryHash) {
        for (a, b) in self.0.iter_mut().zip(entry.0.iter()) {
            *a = a.wrapping_sub(*b);
        }
    }

    /// Update a KV entry: remove the old hash and add the new one.
    /// This is O(1) regardless of total state size.
    #[inline]
    pub fn update(&mut self, old: &EntryHash, new: &EntryHash) {
        for i in 0..WORLD_HASH_LEN {
            self.0[i] = self.0[i].wrapping_sub(old.0[i]).wrapping_add(new.0[i]);
        }
    }

    /// Merge another WorldHash into this one (commutative).
    /// Used for parallel delta reduction.
    #[inline]
    pub fn merge(&mut self, other: &WorldHash) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = a.wrapping_add(*b);
        }
    }

    /// Subtract another WorldHash (for delta removal in parallel reduce).
    #[inline]
    pub fn subtract(&mut self, other: &WorldHash) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = a.wrapping_sub(*b);
        }
    }

    /// Compute the 32-byte state root: `BLAKE3(world_hash_bytes)`.
    pub fn state_root(&self) -> B256 {
        let bytes = self.to_bytes();
        let hash = blake3::hash(&bytes);
        B256::from_slice(hash.as_bytes())
    }

    /// Serialize world hash to 2048 bytes for persistence.
    pub fn to_bytes(&self) -> [u8; WORLD_HASH_BYTES] {
        let mut bytes = [0u8; WORLD_HASH_BYTES];
        for (i, &val) in self.0.iter().enumerate() {
            let le = val.to_le_bytes();
            bytes[i * 2] = le[0];
            bytes[i * 2 + 1] = le[1];
        }
        bytes
    }

    /// Deserialize world hash from 2048 bytes.
    pub fn from_bytes(bytes: &[u8; WORLD_HASH_BYTES]) -> Self {
        let mut elements = [0u16; WORLD_HASH_LEN];
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            elements[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Self(elements)
    }
}

/// A state change to be applied to the world hash.
#[derive(Debug, Clone)]
pub enum StateChange {
    /// Insert or update an account.
    /// `old` is `None` if this is a new account.
    Account {
        addr: Address,
        old: Option<AccountState>,
        new: AccountState,
    },
    /// Delete an account.
    DeleteAccount {
        addr: Address,
        old: AccountState,
    },
    /// Insert or update a storage slot.
    /// `old_value` is `U256::ZERO` if the slot was previously empty.
    Storage {
        addr: Address,
        slot: B256,
        old_value: U256,
        new_value: U256,
    },
}

/// Account state fields relevant to LtHash commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountState {
    pub nonce: u64,
    pub balance: U256,
    pub code_hash: B256,
}

/// Apply a batch of state changes to a world hash.
///
/// Changes are applied sequentially. For parallel computation, use
/// [`apply_parallel`] instead.
pub fn apply_sequential(world: &mut WorldHash, changes: &[StateChange]) {
    for change in changes {
        apply_one(world, change);
    }
}

fn apply_one(world: &mut WorldHash, change: &StateChange) {
    match change {
        StateChange::Account { addr, old, new } => {
            let new_hash =
                EntryHash::for_account(addr, new.nonce, new.balance, new.code_hash);
            if let Some(old) = old {
                let old_hash =
                    EntryHash::for_account(addr, old.nonce, old.balance, old.code_hash);
                world.update(&old_hash, &new_hash);
            } else {
                world.add(&new_hash);
            }
        }
        StateChange::DeleteAccount { addr, old } => {
            let old_hash =
                EntryHash::for_account(addr, old.nonce, old.balance, old.code_hash);
            world.remove(&old_hash);
        }
        StateChange::Storage { addr, slot, old_value, new_value } => {
            if old_value == new_value {
                return;
            }
            if old_value.is_zero() {
                // New slot
                let new_hash = EntryHash::for_storage(addr, *slot, *new_value);
                world.add(&new_hash);
            } else if new_value.is_zero() {
                // Deleted slot
                let old_hash = EntryHash::for_storage(addr, *slot, *old_value);
                world.remove(&old_hash);
            } else {
                let old_hash = EntryHash::for_storage(addr, *slot, *old_value);
                let new_hash = EntryHash::for_storage(addr, *slot, *new_value);
                world.update(&old_hash, &new_hash);
            }
        }
    }
}

/// Apply a batch of state changes to a world hash using rayon for parallelism.
///
/// Each change is hashed in parallel, and the resulting deltas are reduced
/// into the world hash. Addition is commutative so no locking is needed.
pub fn apply_parallel(world: &mut WorldHash, changes: &[StateChange]) {
    // Compute per-change delta WorldHash in parallel
    let delta = changes
        .par_iter()
        .map(|change| {
            let mut delta = WorldHash::new();
            apply_one(&mut delta, change);
            delta
        })
        .reduce(WorldHash::new, |mut acc, d| {
            acc.merge(&d);
            acc
        });

    world.merge(&delta);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::address;

    fn dummy_addr() -> Address {
        address!("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    }

    #[test]
    fn test_empty_state_root_is_deterministic() {
        let w1 = WorldHash::new();
        let w2 = WorldHash::new();
        assert_eq!(w1.state_root(), w2.state_root());
    }

    #[test]
    fn test_add_remove_is_identity() {
        let mut world = WorldHash::new();
        let empty_root = world.state_root();

        let entry = EntryHash::for_account(
            &dummy_addr(),
            1,
            U256::from(100),
            B256::ZERO,
        );
        world.add(&entry);
        assert_ne!(world.state_root(), empty_root);

        world.remove(&entry);
        assert_eq!(world.state_root(), empty_root);
    }

    #[test]
    fn test_commutativity() {
        let addr1 = address!("1111111111111111111111111111111111111111");
        let addr2 = address!("2222222222222222222222222222222222222222");

        let e1 = EntryHash::for_account(&addr1, 1, U256::from(100), B256::ZERO);
        let e2 = EntryHash::for_account(&addr2, 2, U256::from(200), B256::ZERO);

        let mut w1 = WorldHash::new();
        w1.add(&e1);
        w1.add(&e2);

        let mut w2 = WorldHash::new();
        w2.add(&e2);
        w2.add(&e1);

        assert_eq!(w1.state_root(), w2.state_root());
    }

    #[test]
    fn test_update_is_remove_then_add() {
        let addr = dummy_addr();
        let old_state = AccountState { nonce: 1, balance: U256::from(100), code_hash: B256::ZERO };
        let new_state = AccountState { nonce: 2, balance: U256::from(200), code_hash: B256::ZERO };

        // Method 1: update
        let mut w1 = WorldHash::new();
        let old_hash = EntryHash::for_account(&addr, old_state.nonce, old_state.balance, old_state.code_hash);
        w1.add(&old_hash);
        let new_hash = EntryHash::for_account(&addr, new_state.nonce, new_state.balance, new_state.code_hash);
        w1.update(&old_hash, &new_hash);

        // Method 2: remove then add
        let mut w2 = WorldHash::new();
        w2.add(&old_hash);
        w2.remove(&old_hash);
        w2.add(&new_hash);

        assert_eq!(w1.state_root(), w2.state_root());
    }

    #[test]
    fn test_sequential_parallel_equivalence() {
        let changes: Vec<StateChange> = (0..100)
            .map(|i| {
                let addr = Address::from_slice(&[i as u8; 20]);
                StateChange::Account {
                    addr,
                    old: None,
                    new: AccountState {
                        nonce: i as u64,
                        balance: U256::from(i * 1000u64),
                        code_hash: B256::ZERO,
                    },
                }
            })
            .collect();

        let mut w_seq = WorldHash::new();
        apply_sequential(&mut w_seq, &changes);

        let mut w_par = WorldHash::new();
        apply_parallel(&mut w_par, &changes);

        assert_eq!(w_seq.state_root(), w_par.state_root());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut world = WorldHash::new();
        let entry = EntryHash::for_account(&dummy_addr(), 42, U256::from(9999), B256::ZERO);
        world.add(&entry);

        let bytes = world.to_bytes();
        let restored = WorldHash::from_bytes(&bytes);
        assert_eq!(world, restored);
        assert_eq!(world.state_root(), restored.state_root());
    }
}
