pub mod analysis;
pub mod defensibility;
pub mod types;
mod verification_test;

#[cfg(feature = "wasm")]
use crate::analysis::{decompose_inner, optimize_inner};
#[cfg(feature = "wasm")]
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

#[cfg(all(test, target_arch = "wasm32", feature = "wasm"))]
mod tests {
    use super::*;
    use crate::types::{AllocationStrategy, OptimizationRequest, OptimizationTarget, RangeTarget};
    use wasm_bindgen::JsValue;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_optimize_invalid_js_value() {
        // Attempt to optimize with an invalid JsValue (e.g., a simple string instead of an object)
        let invalid_val = JsValue::from_str("invalid_request_data");

        let result = optimize(invalid_val);

        // It should fail to deserialize into an OptimizationRequest
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_optimize_valid_request() {
        // Create mock data
        let mut csv_data = String::new();
        csv_data.push_str("wage,group,education,experience\n");
        for i in 0..10 {
            let ed = 12 + (i % 4);
            let ex = 5 + (i % 6);
            csv_data.push_str(&format!("50000,A,{},{}\n", ed, ex));
            csv_data.push_str(&format!("40000,B,{},{}\n", ed, ex));
        }

        let req = OptimizationRequest {
            csv_data: csv_data.into_bytes(),
            outcome_variable: "wage".to_string(),
            group_variable: "group".to_string(),
            reference_group: "A".to_string(),
            predictors: vec!["education".to_string(), "experience".to_string()],
            categorical_predictors: None,
            budget: 100000.0,
            target_gap: None,
            target: Some(OptimizationTarget::Reference),
            strategy: Some(AllocationStrategy::Greedy),
            min_gap_pct: None,
            forensic_mode: Some(false),
            adjust_both_groups: Some(false),
            confidence_level: None,
            range_target: Some(RangeTarget::Midpoint),
        };

        // Serialize into JsValue
        let valid_val = serde_wasm_bindgen::to_value(&req)
            .expect("Failed to serialize OptimizationRequest to JsValue");

        // Call the WASM wrapper
        let result = optimize(valid_val);

        // It should succeed and return a JsValue representing OptimizationResult
        assert!(result.is_ok());

        // We could also deserialize the result back to verify the wrapper
        let res_val = result.unwrap();
        let opt_res: crate::types::OptimizationResult = serde_wasm_bindgen::from_value(res_val)
            .expect("Failed to deserialize OptimizationResult from JsValue");

        // Verify a basic property to ensure it ran successfully
        assert!(opt_res.total_cost >= 0.0);
    }
}
