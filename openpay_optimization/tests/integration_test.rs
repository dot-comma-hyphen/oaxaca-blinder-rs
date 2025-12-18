#[cfg(test)]
mod tests {
    #[cfg(feature = "pay_equity")]
    use oaxaca_blinder::OaxacaBuilder;
    use openpay_optimization::engine::OptimizationEngine;
    #[cfg(feature = "pay_equity")]
    use openpay_optimization::pay_equity::PayEquityProblem;
    use openpay_optimization::wage_scale::WageScaleProblem;
    use polars::prelude::*;

    #[cfg(feature = "pay_equity")]
    #[test]
    fn test_pay_equity_remediation() -> anyhow::Result<()> {
        // Create dummy data
        // Group A: High salary, High Education
        // Group B: Low salary, Same Education (should be adjusted)

        let n = 20;
        let mut wage = Vec::new();
        let mut education = Vec::new();
        let mut gender = Vec::new();

        for i in 0..n {
            if i < 10 {
                // Group A (Reference/Advantaged)
                wage.push(100.0 + i as f64);
                // Vary education to avoid singularity
                education.push(12.0 + (i % 3) as f64);
                gender.push("M");
            } else {
                // Group B
                wage.push(80.0 + (i - 10) as f64);
                education.push(12.0 + (i % 3) as f64); // Same education distribution
                gender.push("F");
            }
        }

        let df = df!(
            "wage" => wage,
            "education" => education,
            "gender" => gender
        )?;

        // Initial Oaxaca
        let mut builder = OaxacaBuilder::new(df.clone(), "wage", "gender", "F");
        builder.predictors(&["education"]);

        let problem = PayEquityProblem::new(builder.clone(), 0.0); // Target 0 gap

        let engine = OptimizationEngine::new();
        let result = engine.solve(problem)?;

        println!("Objective (Total Cost): {}", result.objective_value);

        // Check adjustments
        let mut total_adj = 0.0;
        for (k, v) in &result.solution {
            if k.starts_with("adj_") {
                total_adj += v;
                // Assert adjustments are non-negative
                assert!(*v >= -1e-6);
            }
        }

        assert!(total_adj > 0.0, "Should require adjustment to close gap");

        let map_sum: f64 = result.solution.values().sum();
        println!("Map Sum: {}", map_sum);

        assert!(map_sum > 150.0);

        Ok(())
    }

    #[test]
    fn test_wage_scale_optimization() -> anyhow::Result<()> {
        // Census: 2 people.
        // P1: Grade 1, Step 1, Salary 30000.
        // P2: Grade 2, Step 1, Salary 40000.

        let df = df!(
            "grade" => &[1u32, 2u32],
            "step" => &[1u32, 1u32],
            "salary" => &[30000.0, 40000.0]
        )?;

        // Budget large enough
        let budget = 100_000.0;
        let min_wage = 25_000.0;
        let problem = WageScaleProblem::new(df, budget, 2, 2, min_wage);

        let engine = OptimizationEngine::new();
        let result = engine.solve(problem)?;

        // Verify constraints
        let s11 = result.solution["step_1_1"];
        let s12 = result.solution["step_1_2"];
        let s21 = result.solution["step_2_1"];

        assert!(s11 >= 30000.0); // No pay cut
        assert!(s21 >= 40000.0); // No pay cut

        // Verify empty cell (1, 2) respects structure
        assert!(s12 >= s11 * 1.03 - 1e-4);

        // Verify min wage (implicit since 30k > 25k, but crucial for logic check)
        assert!(s11 >= min_wage);

        println!("Wage Scale: {:?}", result.solution);

        Ok(())
    }

    #[test]
    fn test_merit_matrix_optimization() -> anyhow::Result<()> {
        use openpay_optimization::merit_matrix::MeritMatrixProblem;

        // 3 Employees
        // E1: Rating 5.0 (Top), Salary 100k
        // E2: Rating 3.0 (Mid), Salary 100k
        // E3: Rating 1.0 (Low), Salary 100k
        let df = df!(
            "performance_rating" => &[5.0, 3.0, 1.0],
            "salary" => &[100_000.0, 100_000.0, 100_000.0]
        )?;

        // Budget: 10k total (enough for 10% on one person, or split)
        // Since E1 has highest rating, optimizer should maximize E1's increase.
        let budget = 10_000.0;
        let problem = MeritMatrixProblem::new(df, budget);

        let engine = OptimizationEngine::new();
        let result = engine.solve(problem)?;

        let inc0 = result.solution["increase_pct_0"]; // E1
        let inc1 = result.solution["increase_pct_1"]; // E2
        let inc2 = result.solution["increase_pct_2"]; // E3

        println!(
            "Merit Increases: E1={:.4}, E2={:.4}, E3={:.4}",
            inc0, inc1, inc2
        );

        // Expect E1 to get max increase (10%) because 5.0 > 3.0
        assert!((inc0 - 0.10).abs() < 1e-4);

        // E2 and E3 should get 0 because budget is exhausted by E1 (10% * 100k = 10k)
        assert!(inc1 < 1e-4);
        assert!(inc2 < 1e-4);

        Ok(())
    }
}
