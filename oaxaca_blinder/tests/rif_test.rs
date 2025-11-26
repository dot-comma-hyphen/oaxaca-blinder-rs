use polars::prelude::*;
use oaxaca_blinder::OaxacaBuilder;

#[test]
fn test_rif_decomposition() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dataset where Group B has higher variance than Group A
    // Group A (M): Mean ~22, High Variance
    // Group B (F): Mean ~22, Low Variance
    
    let mut wage = Vec::new();
    let mut group = Vec::new();
    let mut education = Vec::new();

    // Group B (Reference: "F") - Low Variance
    for i in 0..100 {
        wage.push(20.0 + (i as f64 % 5.0)); // Range 20-24
        group.push("F");
        education.push(12.0 + (i % 4) as f64);
    }

    // Group A (Comparison: "M") - High Variance
    for i in 0..100 {
        wage.push(15.0 + (i as f64 % 15.0)); // Range 15-29
        group.push("M");
        education.push(12.0 + (i % 4) as f64);
    }

    let df = df!(
        "wage" => wage,
        "group" => group,
        "education" => education
    )?;

    // Decompose at the 90th percentile
    let results = OaxacaBuilder::new(df, "wage", "group", "F")
        .predictors(&["education"])
        .bootstrap_reps(10)
        .decompose_quantile(0.9)?;

    results.summary();

    // At 0.9 quantile:
    // Group M (A) should be significantly higher than Group F (B)
    // because M has higher variance and thus a higher upper tail.
    // Gap = Y_A - Y_B
    
    println!("Total Gap at Q90: {}", results.total_gap());
    
    // Gap should be positive
    assert!(*results.total_gap() > 0.0);

    Ok(())
}
