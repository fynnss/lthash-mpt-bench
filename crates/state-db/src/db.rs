//! RocksDB state database implementation.

use crate::error::{DbError, Result};
use alloy_primitives::{Address, B256, U256};
use lthash::{
    AccountState, StateChange, WorldHash, WORLD_HASH_BYTES,
    apply_sequential, apply_parallel,
};
use rocksdb::{
    ColumnFamilyDescriptor, Options, WriteBatch, DB,
};
use std::path::Path;

// Column family names
const CF_ACCOUNTS: &str = "accounts";
const CF_STORAGES: &str = "storages";
const CF_BYTECODES: &str = "bytecodes";
const CF_META: &str = "meta";

// Meta keys
const META_WORLD_HASH: &[u8] = b"world_hash";

/// Flat KV state database backed by RocksDB with LtHash commitment.
pub struct StateDb {
    db: DB,
    world_hash: WorldHash,
}

impl StateDb {
    /// Open (or create) a StateDb at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_opts = Options::default();
        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_ACCOUNTS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_STORAGES, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_BYTECODES, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_META, cf_opts),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)?;

        // Load existing world hash from meta CF, or start fresh.
        let world_hash = {
            let cf_meta = db.cf_handle(CF_META).expect("meta CF always exists");
            match db.get_cf(&cf_meta, META_WORLD_HASH)? {
                Some(bytes) => {
                    if bytes.len() != WORLD_HASH_BYTES {
                        return Err(DbError::CorruptWorldHash {
                            expected: WORLD_HASH_BYTES,
                            got: bytes.len(),
                        });
                    }
                    let arr: &[u8; WORLD_HASH_BYTES] =
                        bytes.as_slice().try_into().unwrap();
                    WorldHash::from_bytes(arr)
                }
                None => WorldHash::new(),
            }
        };

        Ok(Self { db, world_hash })
    }

    /// Current 32-byte state root.
    pub fn state_root(&self) -> B256 {
        self.world_hash.state_root()
    }

    /// Apply a batch of state changes atomically.
    ///
    /// - Updates the in-memory `WorldHash` (sequential).
    /// - Writes all KV changes and the new world hash to RocksDB atomically.
    pub fn apply(&mut self, changes: &[StateChange]) -> Result<()> {
        apply_sequential(&mut self.world_hash, changes);
        self.flush_changes(changes)?;
        Ok(())
    }

    /// Apply a batch of state changes using rayon for parallel hash computation.
    ///
    /// The hash computation is parallelised; the RocksDB write is still a
    /// single atomic batch.
    pub fn apply_parallel(&mut self, changes: &[StateChange]) -> Result<()> {
        apply_parallel(&mut self.world_hash, changes);
        self.flush_changes(changes)?;
        Ok(())
    }

    /// Write all KV changes + updated world hash to RocksDB in one batch.
    fn flush_changes(&self, changes: &[StateChange]) -> Result<()> {
        let cf_accounts = self.db.cf_handle(CF_ACCOUNTS).unwrap();
        let cf_storages = self.db.cf_handle(CF_STORAGES).unwrap();
        let cf_meta = self.db.cf_handle(CF_META).unwrap();

        let mut batch = WriteBatch::default();

        for change in changes {
            match change {
                StateChange::Account { addr, new, .. } => {
                    let key = addr.as_slice();
                    let value = encode_account(new);
                    batch.put_cf(&cf_accounts, key, value);
                }
                StateChange::DeleteAccount { addr, .. } => {
                    batch.delete_cf(&cf_accounts, addr.as_slice());
                }
                StateChange::Storage { addr, slot, new_value, .. } => {
                    let key = encode_storage_key(addr, slot);
                    if new_value.is_zero() {
                        batch.delete_cf(&cf_storages, key);
                    } else {
                        batch.put_cf(&cf_storages, key, new_value.to_be_bytes::<32>());
                    }
                }
            }
        }

        // Persist the updated world hash
        batch.put_cf(&cf_meta, META_WORLD_HASH, self.world_hash.to_bytes());

        self.db.write(batch)?;
        Ok(())
    }

    // ── Read API ─────────────────────────────────────────────────────────────

    /// Batch read accounts via RocksDB multi_get (one round-trip).
    pub fn multi_get_accounts(&self, addrs: &[Address]) -> Result<Vec<Option<AccountState>>> {
        let cf = self.db.cf_handle(CF_ACCOUNTS).unwrap();
        let keys: Vec<(&rocksdb::ColumnFamily, &[u8])> =
            addrs.iter().map(|a| (cf, a.as_slice())).collect();
        self.db
            .multi_get_cf(keys)
            .into_iter()
            .map(|r| match r? {
                None => Ok(None),
                Some(bytes) => Ok(Some(decode_account(&bytes)?)),
            })
            .collect()
    }

    /// Batch read storage slots via RocksDB multi_get (one round-trip).
    pub fn multi_get_storage(
        &self,
        keys: &[(Address, B256)],
    ) -> Result<Vec<U256>> {
        let cf = self.db.cf_handle(CF_STORAGES).unwrap();
        let encoded: Vec<[u8; 52]> =
            keys.iter().map(|(a, s)| encode_storage_key(a, s)).collect();
        let cf_keys: Vec<(&rocksdb::ColumnFamily, &[u8])> =
            encoded.iter().map(|k| (cf, k.as_slice())).collect();
        self.db
            .multi_get_cf(cf_keys)
            .into_iter()
            .map(|r| match r? {
                None => Ok(U256::ZERO),
                Some(bytes) => {
                    if bytes.len() != 32 {
                        return Err(DbError::CorruptStorage(bytes.len()));
                    }
                    Ok(U256::from_be_slice(&bytes))
                }
            })
            .collect()
    }

    /// Read account state.
    pub fn get_account(&self, addr: &Address) -> Result<Option<AccountState>> {
        let cf = self.db.cf_handle(CF_ACCOUNTS).unwrap();
        match self.db.get_cf(&cf, addr.as_slice())? {
            None => Ok(None),
            Some(bytes) => Ok(Some(decode_account(&bytes)?)),
        }
    }

    /// Read a storage slot value.
    pub fn get_storage(&self, addr: &Address, slot: &B256) -> Result<U256> {
        let cf = self.db.cf_handle(CF_STORAGES).unwrap();
        let key = encode_storage_key(addr, slot);
        match self.db.get_cf(&cf, key)? {
            None => Ok(U256::ZERO),
            Some(bytes) => {
                if bytes.len() != 32 {
                    return Err(DbError::CorruptStorage(bytes.len()));
                }
                Ok(U256::from_be_slice(&bytes))
            }
        }
    }

    /// Store bytecode (by code hash, content-addressed).
    pub fn put_bytecode(&self, code_hash: &B256, code: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_BYTECODES).unwrap();
        self.db.put_cf(&cf, code_hash.as_slice(), code)?;
        Ok(())
    }

    /// Read bytecode by code hash.
    pub fn get_bytecode(&self, code_hash: &B256) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_BYTECODES).unwrap();
        Ok(self.db.get_cf(&cf, code_hash.as_slice())?)
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

/// Encode account state: `nonce(8B LE) || balance(32B LE) || code_hash(32B)` = 72B
fn encode_account(state: &AccountState) -> [u8; 72] {
    let mut buf = [0u8; 72];
    buf[0..8].copy_from_slice(&state.nonce.to_le_bytes());
    buf[8..40].copy_from_slice(&state.balance.to_le_bytes::<32>());
    buf[40..72].copy_from_slice(state.code_hash.as_slice());
    buf
}

fn decode_account(bytes: &[u8]) -> Result<AccountState> {
    if bytes.len() != 72 {
        return Err(DbError::CorruptAccount(bytes.len()));
    }
    let nonce = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let balance = U256::from_le_slice(&bytes[8..40]);
    let code_hash = B256::from_slice(&bytes[40..72]);
    Ok(AccountState { nonce, balance, code_hash })
}

/// Encode storage key: `addr(20B) || slot(32B)` = 52B
fn encode_storage_key(addr: &Address, slot: &B256) -> [u8; 52] {
    let mut key = [0u8; 52];
    key[0..20].copy_from_slice(addr.as_slice());
    key[20..52].copy_from_slice(slot.as_slice());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::address;
    use tempfile::tempdir;

    fn test_account(nonce: u64, balance: u64) -> AccountState {
        AccountState {
            nonce,
            balance: U256::from(balance),
            code_hash: B256::ZERO,
        }
    }

    #[test]
    fn test_open_fresh_db() {
        let dir = tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        // Empty state root should be deterministic BLAKE3 of zeroed world hash
        let root = db.state_root();
        assert_ne!(root, B256::ZERO); // it's BLAKE3 of 2048 zero bytes, not zero
    }

    #[test]
    fn test_insert_and_read_account() {
        let dir = tempdir().unwrap();
        let mut db = StateDb::open(dir.path()).unwrap();
        let addr = address!("1111111111111111111111111111111111111111");

        let changes = vec![StateChange::Account {
            addr,
            old: None,
            new: test_account(1, 1000),
        }];
        db.apply(&changes).unwrap();

        let acc = db.get_account(&addr).unwrap().unwrap();
        assert_eq!(acc.nonce, 1);
        assert_eq!(acc.balance, U256::from(1000u64));
    }

    #[test]
    fn test_world_hash_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let addr = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

        let root1 = {
            let mut db = StateDb::open(dir.path()).unwrap();
            db.apply(&[StateChange::Account {
                addr,
                old: None,
                new: test_account(5, 9999),
            }])
            .unwrap();
            db.state_root()
        };

        // Reopen and check root is restored
        let db2 = StateDb::open(dir.path()).unwrap();
        assert_eq!(db2.state_root(), root1);
    }

    #[test]
    fn test_storage_insert_read_delete() {
        let dir = tempdir().unwrap();
        let mut db = StateDb::open(dir.path()).unwrap();
        let addr = address!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let slot = B256::from([1u8; 32]);

        // Insert
        db.apply(&[StateChange::Storage {
            addr,
            slot,
            old_value: U256::ZERO,
            new_value: U256::from(42u64),
        }])
        .unwrap();
        assert_eq!(db.get_storage(&addr, &slot).unwrap(), U256::from(42u64));

        // Delete
        db.apply(&[StateChange::Storage {
            addr,
            slot,
            old_value: U256::from(42u64),
            new_value: U256::ZERO,
        }])
        .unwrap();
        assert_eq!(db.get_storage(&addr, &slot).unwrap(), U256::ZERO);
    }
}
