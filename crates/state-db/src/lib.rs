//! RocksDB-backed flat KV state database with LtHash state commitment.
//!
//! ## Storage schema
//!
//! Column families:
//! - `accounts`  — `addr(20B)` → `nonce(8B LE) || balance(32B LE) || code_hash(32B)` = 72B
//! - `storages`  — `addr(20B) || slot(32B)` → `value(32B BE)`
//! - `bytecodes` — `code_hash(32B)` → `bytecode bytes`
//! - `meta`      — `"world_hash"` → `2048B world hash`
//!
//! The LtHash `WorldHash` is kept in memory and flushed to the `meta` CF
//! on every commit. Reads always go directly to RocksDB.

pub mod db;
pub mod error;

pub use db::StateDb;
pub use error::DbError;
pub use lthash::{AccountState, StateChange};
