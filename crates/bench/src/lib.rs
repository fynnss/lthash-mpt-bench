//! Shared utilities for benchmarks.

use alloy_primitives::{Address, B256, U256};
use lthash::AccountState;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A generated account in the base state.
#[derive(Clone)]
pub struct Account {
    pub addr: Address,
    pub nonce: u64,
    pub balance: U256,
}

/// Pre-computed (old, new) pair for one account — simulates the EVM pre-state cache.
#[derive(Clone)]
pub struct AccountDelta {
    pub addr: Address,
    pub old: AccountState,
    pub new: AccountState,
}

/// Generate `n` deterministic random accounts with the given `seed`.
pub fn gen_accounts(n: usize, seed: u64) -> Vec<Account> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| Account {
            addr: Address::from_slice(&rng.gen::<[u8; 20]>()),
            nonce: rng.gen::<u64>() & 0xFFFF,
            balance: U256::from(rng.gen::<u128>()),
        })
        .collect()
}

/// Generate a block touching the first `n` accounts from `base`.
///
/// Returns pre-computed (old, new) pairs — nonce+1, balance incremented by a random u64.
pub fn gen_block(base: &[Account], n: usize) -> Vec<AccountDelta> {
    let mut rng = StdRng::seed_from_u64(42);
    base[..n]
        .iter()
        .map(|a| AccountDelta {
            addr: a.addr,
            old: AccountState { nonce: a.nonce, balance: a.balance, code_hash: B256::ZERO },
            new: AccountState {
                nonce: a.nonce + 1,
                balance: a.balance + U256::from(rng.gen::<u64>()),
                code_hash: B256::ZERO,
            },
        })
        .collect()
}
