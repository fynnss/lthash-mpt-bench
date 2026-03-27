use thiserror::Error;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("Corrupt world hash: expected {expected} bytes, got {got}")]
    CorruptWorldHash { expected: usize, got: usize },

    #[error("Corrupt account value: expected 72 bytes, got {0}")]
    CorruptAccount(usize),

    #[error("Corrupt storage value: expected 32 bytes, got {0}")]
    CorruptStorage(usize),
}

pub type Result<T> = std::result::Result<T, DbError>;
