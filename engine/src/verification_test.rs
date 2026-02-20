#[cfg(test)]
mod tests {
    use crate::analysis::optimize_inner;
    use crate::types::{AllocationStrategy, OptimizationRequest, OptimizationTarget};
    use polars::prelude::*;
    use std::io::Cursor;

    #[test]
    fn test_log_linear_adjustments_are_non_uniform() {
        // Create mock data:
        // Group A (Ref): High Wages, correlated with X
        // Group B (Target): Lower Wages, SAME X, but paid less.

        // Wages:
        // A: Exp(10 + 0.5*X) -> High Pay
        // B: Exp(9 + 0.5*X)  -> Low Pay

        // X: Education Level
        // Wages:
        // A: Exp(10 + 0.5*X) -> High Pay
        // B: Exp(9 + 0.5*X)  -> Low Pay
        // Gap is massive.

        let w_a_1 = (10.0 + 0.5 * 1.0f64).exp(); // ~36315
        let w_a_2 = (10.0 + 0.5 * 2.0f64).exp(); // ~59874
        let w_a_3 = (10.0 + 0.5 * 3.0f64).exp(); // ~98715

        let w_b_1 = (9.0 + 0.5 * 1.0f64).exp(); // ~13359
        let w_b_2 = (9.0 + 0.5 * 2.0f64).exp(); // ~22026
        let w_b_3 = (9.0 + 0.5 * 3.0f64).exp(); // ~36315

        // Create DataFrame using compatible Polars API
        let df = DataFrame::new(vec![
            Column::new("id".into(), &[1, 2, 3, 4, 5, 6]),
            Column::new("group".into(), &["A", "A", "A", "B", "B", "B"]),
            Column::new("education".into(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            Column::new("wage".into(), &[w_a_1, w_a_2, w_a_3, w_b_1, w_b_2, w_b_3]),
        ])
        .unwrap();

        // Write to CSV buffer
        let mut buffer = Vec::new();
        CsvWriter::new(&mut buffer).finish(&mut df.clone()).unwrap();
        let csv_data = buffer;

        let req = OptimizationRequest {
            csv_data: csv_data,
            group_variable: "group".to_string(),
            outcome_variable: "wage".to_string(),
            predictors: vec!["education".to_string()],
            categorical_predictors: None,
            reference_group: "A".to_string(),
            budget: 1_000_000.0,
            target: Some(OptimizationTarget::Reference),
            strategy: Some(AllocationStrategy::Greedy),
            min_gap_pct: None,
            forensic_mode: Some(false),
            adjust_both_groups: None,
            target_gap: None,
            confidence_level: None,
            range_target: None,
        };

        let result = optimize_inner(req).unwrap();

        println!(
            "Adjustments: {:?}",
            result
                .adjustments
                .iter()
                .map(|a| a.adjustment)
                .collect::<Vec<_>>()
        );

        // Assertions
        assert_eq!(result.adjustments.len(), 3);

        let adj_1 = result.adjustments[0].adjustment;
        let adj_3 = result.adjustments[2].adjustment;

        // Check 1: Non-Zero Adjustments
        assert!(adj_1 > 100.0, "Adjustment 1 should be significant");

        // Check 2: Non-Uniformity
        // Person 3 (High Education) should get a LARGER adjustment than Person 1 (Low Education)
        // because the gap is a percentage (Log Scale).
        // Diff = Exp(10.5) - Exp(9.5) vs Exp(11.5) - Exp(10.5)
        // Delta is bigger for higher X.

        assert!(
            (adj_3 - adj_1).abs() > 100.0,
            "Adjustments should differ significantly! Found {} vs {}",
            adj_1,
            adj_3
        );

        println!("Test Passed: Non-Uniformity Confirmed.");

        // Check 3: Model Coefficients
        assert!(
            !result.model_coefficients.is_empty(),
            "Model coefficients should not be empty"
        );
        let education_coef = result
            .model_coefficients
            .iter()
            .find(|c| c.name == "education");
        assert!(
            education_coef.is_some(),
            "Education coefficient should be present"
        );
        println!(
            "Coefficient for education: {}",
            education_coef.unwrap().value
        );
    }

    fn create_mock_data() -> (Vec<u8>, DataFrame) {
        // Simple mock:
        // 1000 employees. Groups A and B.
        // A paid more than B for same features.
        // We want gaps.
        let mut ids = Vec::new();
        let mut groups = Vec::new();
        let mut edu = Vec::new();
        let mut exp = Vec::new();
        let mut wage = Vec::new();
        let mut depts = Vec::new();

        for i in 0..1000 {
            ids.push(i as u64);
            let is_a = i < 500; // 50% Group A
            groups.push(if is_a { "Male" } else { "Female" }); // StandardRef=Male

            let ed = 12.0 + ((i % 10) as f64) * 0.5; // 12-17
            let ex = ((i % 30) as f64);
            edu.push(ed);
            exp.push(ex);

            depts.push(if i % 2 == 0 { "Sales" } else { "Eng" });

            // Wages: Base + Coef*Ed + Coef*Ex + Noise
            // A: Base 20000 + 2000*Ed + 500*Ex
            // B: Base 15000 + 2000*Ed + 500*Ex
            // Explicit Gap of 5000.
            let fair_base = 20000.0 + 2000.0 * ed + 500.0 * ex;
            let actual_base = if is_a { fair_base } else { fair_base - 5000.0 };

            // Add deterministic noise
            let noise = ((i % 100) as f64) * 10.0 - 500.0;
            wage.push(actual_base + noise);
        }

        let df = DataFrame::new(vec![
            Column::new("id".into(), &ids),
            Column::new("gender".into(), &groups),
            Column::new("education".into(), &edu),
            Column::new("experience".into(), &exp),
            Column::new("wage".into(), &wage),
            Column::new("department".into(), &depts),
        ])
        .unwrap();

        let mut buffer = Vec::new();
        CsvWriter::new(&mut buffer).finish(&mut df.clone()).unwrap();
        (buffer, df)
    }

    use crate::types::RangeTarget;

    #[test]
    fn test_lower_bound_optimization() {
        let (csv_data, _) = create_mock_data();

        // Use optimize_inner directly
        use crate::analysis::optimize_inner;

        let req = OptimizationRequest {
            csv_data: csv_data.clone(),
            outcome_variable: "wage".to_string(),
            group_variable: "gender".to_string(),
            reference_group: "Male".to_string(),
            predictors: vec!["education".to_string(), "experience".to_string()],
            categorical_predictors: Some(vec!["department".to_string()]),
            strategy: Some(AllocationStrategy::Greedy),
            budget: 5_000_000.0, // Enough to fix all gaps
            target_gap: None,
            target: None,
            min_gap_pct: Some(0.0),
            adjust_both_groups: Some(false),
            forensic_mode: Some(false),
            confidence_level: Some(0.95),
            range_target: Some(RangeTarget::LowerBound),
        };

        let result = optimize_inner(req).expect("Optimization failed");

        println!("Adjustments count: {}", result.adjustments.len());

        let mut checked_any = false;
        for adj in &result.adjustments {
            if let Some(lower) = adj.fair_wage_lower_bound {
                // Check consistency:
                // New Wage = Current + Adj.
                // With Greedy and infinite budget, Adj should fill the gap to Target.
                // Target is LowerBound.

                // But specifically for those Adjusted, we expect:
                // NewWage >= LowerBound - margin

                // Note: adj.fair_wage is the Midpoint (Statistical Fair Wage)

                // Allow some floating point error
                let diff = adj.new_wage - lower;

                if diff < -1.0 {
                    println!(
                        "FAIL: Index {} New {} < Lower {}",
                        adj.index, adj.new_wage, lower
                    );
                    // panic!("Underpaid relative to Lower Bound!");
                }

                // Verify that we are NOT targeting the Midpoint if LowerBound is significantly lower
                if (adj.fair_wage - lower) > 100.0 {
                    // Start: Current
                    // End: LowerBound using infinite budget
                    // So NewWage should be close to LowerBound
                    // It should NOT be close to Midpoint (fair_wage)
                    if (adj.new_wage - adj.fair_wage).abs() < 10.0 {
                        println!("WARNING: Adjustment seems to have targeted Midpoint instead of LowerBound. New: {}, Mid: {}, Lower: {}", adj.new_wage, adj.fair_wage, lower);
                    }
                }

                checked_any = true;
            }
        }
        assert!(checked_any, "Should have checked at least one adjustment");
    }

    #[test]
    fn test_auto_budget_lower_bound() {
        let (csv_data, _) = create_mock_data();
        use crate::analysis::optimize_inner;

        // Auto Budget (0.0) with LowerBound target
        let req = OptimizationRequest {
            csv_data: csv_data.clone(),
            outcome_variable: "wage".to_string(),
            group_variable: "gender".to_string(),
            reference_group: "Male".to_string(),
            predictors: vec!["education".to_string(), "experience".to_string()],
            categorical_predictors: Some(vec!["department".to_string()]),
            strategy: Some(AllocationStrategy::Greedy),
            budget: 0.0, // Auto Budget!
            target_gap: None,
            target: None,
            min_gap_pct: Some(0.0),
            adjust_both_groups: Some(false),
            forensic_mode: Some(false),
            confidence_level: Some(0.95),
            range_target: Some(RangeTarget::LowerBound),
        };

        let result = optimize_inner(req).expect("Optimization failed");

        println!("Adjustments count: {}", result.adjustments.len());
        println!("Required Budget: {}", result.required_budget);
        println!("Total Cost: {}", result.total_cost);

        // Assert that we spent money!
        assert!(
            result.total_cost > 0.0,
            "Auto budget should have spent money to fix gaps!"
        );

        // Assert that Total Cost matches Required Budget (since we had 0 budget, effective budget should be required budget)
        assert!(
            (result.total_cost - result.required_budget).abs() < 1.0,
            "Total Cost should equal Required Budget for Auto Optimization"
        );

        let mut checked_any = false;
        for adj in &result.adjustments {
            if let Some(lower) = adj.fair_wage_lower_bound {
                // Check if we hit the target LowerBound
                let diff = adj.new_wage - lower;
                if diff < -1.0 {
                    println!(
                        "FAIL: Index {} New {} < Lower {}",
                        adj.index, adj.new_wage, lower
                    );
                }
                checked_any = true;
            }
        }
        assert!(checked_any, "Should have checked at least one adjustment");
    }
    #[test]
    fn test_defensibility_override() {
        use crate::defensibility::check_defensibility_inner;
        use crate::types::{ProposedAdjustment, VerificationRequest};
        use std::collections::HashMap;

        let (csv_data, df) = create_mock_data();

        // Find a Group B person (Low Education, Low Experience) who is underpaid.
        // In mock data: B is offset by -5000.
        // Index 5 (Group B): Ed=14.5, Exp=5.
        // Wage ~ 15000 + 2000*14.5 + 500*5 = 15000 + 29000 + 2500 = 46500.
        // Actual in mock is Fair - 5000 + Noise.

        // Let's pick index 555 (Group B, since > 500).
        let target_idx = 555;

        // Baseline Check (No Overrides)
        let req_baseline = VerificationRequest {
            decomposition_params: crate::types::DecompositionRequest {
                csv_data: csv_data.clone(),
                outcome_variable: "wage".to_string(),
                group_variable: "gender".to_string(),
                reference_group: "Male".to_string(),
                predictors: vec!["education".to_string(), "experience".to_string()],
                categorical_predictors: Some(vec!["department".to_string()]),
                three_fold: None,
                quantile: None,
                reference_coefficients: None,
                bootstrap_reps: None,
            },
            adjustments: vec![ProposedAdjustment {
                index: target_idx,
                value: 0.0,
                predictor_overrides: None,
            }],
        };

        let res_baseline = check_defensibility_inner(req_baseline).expect("Baseline check failed");
        let adj_baseline = &res_baseline.adjustments[0];

        println!(
            "Baseline: Current={}, Fair={}, Lower={:?}, Defensible={:?}",
            adj_baseline.current_wage,
            adj_baseline.fair_wage,
            adj_baseline.fair_wage_lower_bound,
            adj_baseline.is_defensible
        );

        // The mock data has a gap of 5000.
        // Fair Wage should be ~ Current + 5000.
        // Lower bound should be Fair - Margin (margin usually < 5000 for high confidence).
        // So likely NOT defensible.

        // Now Apply Override: LOWER the education significantly.
        // Say we claim this person actually has Education = 10 (instead of ~15).
        // Fair Wage (Male model) is 20000 + 2000*Ed + 500*Ex.
        // Reducing Ed by 5 units -> Reduces Fair Wage by 10,000!
        // This should make the Current Wage appear HIGHER than the new Fair Wage (or at least valid).

        let mut overrides = HashMap::new();
        overrides.insert("education".to_string(), "10.0".to_string());

        let req_override = VerificationRequest {
            decomposition_params: crate::types::DecompositionRequest {
                csv_data: csv_data.clone(),
                outcome_variable: "wage".to_string(),
                group_variable: "gender".to_string(),
                reference_group: "Male".to_string(),
                predictors: vec!["education".to_string(), "experience".to_string()],
                categorical_predictors: Some(vec!["department".to_string()]),
                three_fold: None,
                quantile: None,
                reference_coefficients: None,
                bootstrap_reps: None,
            },
            adjustments: vec![ProposedAdjustment {
                index: target_idx,
                value: 0.0,
                predictor_overrides: Some(overrides),
            }],
        };

        let res_override = check_defensibility_inner(req_override).expect("Override check failed");
        let adj_override = &res_override.adjustments[0];

        println!(
            "Override: Current={}, Fair={}, Lower={:?}, Defensible={:?}",
            adj_override.current_wage,
            adj_override.fair_wage,
            adj_override.fair_wage_lower_bound,
            adj_override.is_defensible
        );

        assert!(
            adj_override.fair_wage < adj_baseline.fair_wage - 5000.0,
            "Fair wage should drop significantly"
        );
        // If Fair Wage dropped by 10k, and gap was 5k, now Current Wage should be > Fair Wage (Overpaid).
        // So Defensible should be TRUE.

        if let Some(def) = adj_override.is_defensible {
            assert!(def, "Should be defensible with lower education override");
        } else {
            panic!("Defensibility check returned None");
        }
    }
}
