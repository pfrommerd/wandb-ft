use std::path::PathBuf;

use crate::error::ApiError;

const WANDB_MACHINE: &str = "api.wandb.ai";

/// Locate the netrc file: honor `$NETRC`, otherwise `~/.netrc`.
fn netrc_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("NETRC") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".netrc"))
}

/// Read the wandb API key (the `password` for `machine api.wandb.ai`) from netrc.
///
/// Returns [`ApiError::MissingApiKey`] if the file, the `api.wandb.ai` entry, or
/// its password cannot be found.
pub fn read_api_key() -> Result<String, ApiError> {
    let path = netrc_path()
        .ok_or_else(|| ApiError::MissingApiKey("could not determine netrc path".into()))?;

    let contents = std::fs::read_to_string(&path).map_err(|e| {
        ApiError::MissingApiKey(format!("could not read {}: {e}", path.display()))
    })?;

    // Tokenize on whitespace/newlines. netrc is a flat stream of tokens where
    // `machine <name>`, `login <user>`, `password <secret>` come in pairs.
    let mut tokens = contents.split_whitespace();
    let mut in_target = false;
    while let Some(tok) = tokens.next() {
        match tok {
            "machine" => {
                in_target = tokens.next() == Some(WANDB_MACHINE);
            }
            "default" => {
                // `default` acts as a catch-all machine entry.
                in_target = true;
            }
            "password" if in_target => {
                if let Some(secret) = tokens.next() {
                    return Ok(secret.to_string());
                }
            }
            // skip the value token following login/account/etc.
            "login" | "account" | "password" => {
                tokens.next();
            }
            _ => {}
        }
    }

    Err(ApiError::MissingApiKey(format!(
        "no password for machine `{WANDB_MACHINE}` found in {}",
        path.display()
    )))
}
