use anyhow::{bail, Result};
#[cfg(feature = "wasm")]
use gloo_net::http::Request;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartnerStatus {
    Active,
    Suspended,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartnerConfig {
    pub firm_name: String,
    pub logo_url: String,
    pub brand_color: String,
    pub disclaimer_text: String,
    pub status: PartnerStatus,
}

pub type AccessRegistry = HashMap<String, PartnerConfig>;

pub fn hash_code(code: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(code.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

#[cfg(feature = "wasm")]
pub async fn validate_access_code_inner(code: &str, registry_url: &str) -> Result<PartnerConfig> {
    // v2 change to force update
    let hashed_code = hash_code(code);

    let registry: AccessRegistry = Request::get(registry_url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to fetch registry: {}", e))?
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse registry JSON: {}", e))?;

    if let Some(config) = registry.get(&hashed_code) {
        match config.status {
            PartnerStatus::Active => Ok(config.clone()),
            PartnerStatus::Suspended => Ok(config.clone()),
        }
    } else {
        bail!("Invalid access code")
    }
}

#[cfg(not(feature = "wasm"))]
pub async fn validate_access_code_inner(_code: &str, _registry_url: &str) -> Result<PartnerConfig> {
    bail!("Access code validation is only supported in WASM target for now.")
}
