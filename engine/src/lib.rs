pub mod analysis;
pub mod defensibility;
pub mod types;
mod verification_test;

use crate::analysis::{decompose_inner, optimize_inner};
use crate::types::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- Wasm Wrappers ---

// --- Wasm Wrappers ---

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decompose(val: JsValue) -> Result<JsValue, JsValue> {
    let req: DecompositionRequest = serde_wasm_bindgen::from_value(val)?;
    let res = decompose_inner(req).map_err(|e| JsValue::from_str(&e))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn optimize(val: JsValue) -> Result<JsValue, JsValue> {
    let req: OptimizationRequest = serde_wasm_bindgen::from_value(val)?;
    let res = optimize_inner(req).map_err(|e| JsValue::from_str(&e))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn verify_adjustments(val: JsValue) -> Result<JsValue, JsValue> {
    let req: VerificationRequest = serde_wasm_bindgen::from_value(val)?;
    let res = crate::analysis::verify_inner(req).map_err(|e| JsValue::from_str(&e))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn calculate_efficient_frontier(val: JsValue) -> Result<JsValue, JsValue> {
    let req: EfficientFrontierRequest = serde_wasm_bindgen::from_value(val)?;
    let res = crate::analysis::calculate_efficient_frontier_inner(req)
        .map_err(|e| JsValue::from_str(e.as_str()))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn check_defensibility(val: JsValue) -> Result<JsValue, JsValue> {
    let req: VerificationRequest = serde_wasm_bindgen::from_value(val)?;
    let res = crate::defensibility::check_defensibility_inner(req)
        .map_err(|e| JsValue::from_str(e.as_str()))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub async fn validate_access_code(code: String, registry_url: String) -> Result<JsValue, JsValue> {
    let res = crate::access::validate_access_code_inner(&code, &registry_url)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(serde_wasm_bindgen::to_value(&res)?)
}

mod access;
