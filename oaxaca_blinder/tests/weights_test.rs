use polars::prelude::*;
use oaxaca_blinder::OaxacaBuilder;

#[test]
fn test_weighted_decomposition() -> Result<(), Box<dyn std::error::Error>> {
    // Create data
    // Group A: 
    // 1. Outcome 10, x=1, w=1
    // 2. Outcome 10, x=1, w=1
    // 3. Outcome 2,  x=0, w=10 (Heavy weight on low outcome)
    // Unweighted Mean A = (10+10+2)/3 = 7.333
    // Weighted Mean A = (10*1 + 10*1 + 2*10)/12 = 40/12 = 3.333
    
    // Group B:
    // 1. Outcome 5, x=0, w=1
    // 2. Outcome 7, x=1, w=1
    // Mean B = 6.0 (Unweighted and Weighted same since weights are 1)
    
    let df = df!(
        "outcome" => &[10.0, 10.0, 2.0,  5.0, 7.0],
        "group" =>   &["A",  "A",  "A",  "B", "B"],
        "weight" =>  &[1.0,  1.0,  10.0, 1.0, 1.0],
        "x" =>       &[1.0,  1.0,  0.0,  0.0, 1.0]
    )?;
    
    // Unweighted Gap = 7.333 - 6.0 = 1.333
    // Weighted Gap = 3.333 - 6.0 = -2.666
    
    // Run unweighted
    let res_unweighted = OaxacaBuilder::new(df.clone(), "outcome", "group", "B")
        .predictors(&["x"])
        .bootstrap_reps(0) 
        .run()?;
        
    println!("Unweighted Gap: {}", res_unweighted.total_gap());
    assert!((res_unweighted.total_gap() - 1.333).abs() < 0.01);
    
    // Run weighted
    let res_weighted = OaxacaBuilder::new(df, "outcome", "group", "B")
        .predictors(&["x"])
        .weights("weight")
        .bootstrap_reps(0)
        .run()?;
        
    println!("Weighted Gap: {}", res_weighted.total_gap());
    assert!((res_weighted.total_gap() - (-2.666)).abs() < 0.01);
    
    Ok(())
}
