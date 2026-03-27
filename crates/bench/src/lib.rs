//! Shared utilities for benchmarks and tests.

use alloy_primitives::{Address, B256, U256};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A generated state snapshot: a set of (address, nonce, balance, storage slots).
#[derive(Clone)]
pub struct GeneratedState {
    pub accounts: Vec<AccountEntry>,
}

#[derive(Clone)]
pub struct AccountEntry {
    pub addr: Address,
    pub nonce: u64,
    pub balance: U256,
    pub code_hash: B256,
    pub storage: Vec<(B256, U256)>,
}

/// Generate a reproducible random state with `n_accounts` accounts,
/// each with `slots_per_account` storage slots.
pub fn generate_state(
    n_accounts: usize,
    slots_per_account: usize,
    seed: u64,
) -> GeneratedState {
    let mut rng = StdRng::seed_from_u64(seed);
    let accounts = (0..n_accounts)
        .map(|_| {
            let addr = Address::from_slice(&rng.gen::<[u8; 20]>());
            let nonce = rng.gen::<u64>() & 0xFFFF; // keep small
            let balance = U256::from(rng.gen::<u128>());
            let code_hash = if rng.gen_bool(0.1) {
                B256::from(rng.gen::<[u8; 32]>())
            } else {
                B256::ZERO
            };
            let storage = (0..slots_per_account)
                .map(|_| {
                    let slot = B256::from(rng.gen::<[u8; 32]>());
                    let value = U256::from(rng.gen::<u128>());
                    (slot, value)
                })
                .collect();
            AccountEntry { addr, nonce, balance, code_hash, storage }
        })
        .collect();
    GeneratedState { accounts }
}

/// Convert generated state to LtHash StateChanges (all inserts).
pub fn to_lthash_changes(state: &GeneratedState) -> Vec<lthash::StateChange> {
    let mut changes = Vec::new();
    for acc in &state.accounts {
        changes.push(lthash::StateChange::Account {
            addr: acc.addr,
            old: None,
            new: lthash::AccountState {
                nonce: acc.nonce,
                balance: acc.balance,
                code_hash: acc.code_hash,
            },
        });
        for &(slot, value) in &acc.storage {
            changes.push(lthash::StateChange::Storage {
                addr: acc.addr,
                slot,
                old_value: U256::ZERO,
                new_value: value,
            });
        }
    }
    changes
}
