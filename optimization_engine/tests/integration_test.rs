#[cfg(test)]
mod tests {
    use optimization_engine::engine::OptimizationEngine;
    use optimization_engine::pay_equity::PayEquityProblem;
    use optimization_engine::wage_scale::WageScaleProblem;
    use oaxaca_blinder::OaxacaBuilder;
    use polars::prelude::*;

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

        // Gap is roughly 20. N=10. Cost should be around 200.
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
        let problem = WageScaleProblem::new(df, budget, 2, 2);

        let engine = OptimizationEngine::new();
        let result = engine.solve(problem)?;

        // Verify constraints
        let s11 = result.solution["step_1_1"];
        let s12 = result.solution["step_1_2"];
        let s21 = result.solution["step_2_1"];

        assert!(s11 >= 30000.0); // No pay cut
        assert!(s21 >= 40000.0); // No pay cut

        // Structure
        assert!(s12 >= s11 * 1.03 - 1e-4);
        assert!(s21 >= s11 * 1.10 - 1e-4);

        println!("Wage Scale: {:?}", result.solution);

        Ok(())
    }
}
