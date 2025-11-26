use polars::prelude::*;
use oaxaca_blinder::{OaxacaBuilder, ReferenceCoefficients, decompose_changes, run_dfl};

fn create_dummy_data() -> DataFrame {
    df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "experience" => &[5.0, 10.0, 7.0, 12.0, 15.0, 5.0, 10.0, 7.0, 12.0, 15.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap()
}

#[test]
fn test_reference_groups() {
    let df = create_dummy_data();
    
    // Test Cotton
    let mut builder_cotton = OaxacaBuilder::new(df.clone(), "wage", "gender", "F");
    builder_cotton.predictors(&["education", "experience"])
        .reference_coefficients(ReferenceCoefficients::Cotton);
    let results_cotton = builder_cotton.run().expect("Cotton decomposition failed");
        
    assert!(results_cotton.total_gap() > &0.0);
    
    // Test Neumark
    let mut builder_neumark = OaxacaBuilder::new(df.clone(), "wage", "gender", "F");
    builder_neumark.predictors(&["education", "experience"])
        .reference_coefficients(ReferenceCoefficients::Neumark);
    let results_neumark = builder_neumark.run().expect("Neumark decomposition failed");
        
    assert!(results_neumark.total_gap() > &0.0);
}

#[test]
fn test_jmp_decomposition() {
    // Create T1 data
    let df_t1 = df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap();
    
    // Create T2 data (Gap reduced: Women paid more)
    let df_t2 = df!(
        "wage" => &[15.0, 17.0, 16.0, 18.0, 20.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap();
    
    let mut builder_t1 = OaxacaBuilder::new(df_t1, "wage", "gender", "F");
    builder_t1.predictors(&["education"]);
    
    let mut builder_t2 = OaxacaBuilder::new(df_t2, "wage", "gender", "F");
    builder_t2.predictors(&["education"]);
        
    let jmp_results = decompose_changes(&builder_t1, &builder_t2).expect("JMP failed");
    
    jmp_results.summary();
    
    // Gap T1: Mean(M) - Mean(F) = 22.2 - 12.2 = 10.0
    // Gap T2: Mean(M) - Mean(F) = 22.2 - 17.2 = 5.0
    // Total Change = 5.0 - 10.0 = -5.0
    
    assert!((jmp_results.total_change - (-5.0)).abs() < 1e-4);
}

#[test]
fn test_dfl_reweighting() {
    let df = create_dummy_data();
    
    let dfl_results = run_dfl(
        &df,
        "wage",
        "gender",
        "F",
        &["education".to_string(), "experience".to_string()]
    ).expect("DFL failed");
    
    assert_eq!(dfl_results.grid.len(), 100);
    assert_eq!(dfl_results.density_a.len(), 100);
    assert_eq!(dfl_results.density_b.len(), 100);
    assert_eq!(dfl_results.density_b_counterfactual.len(), 100);
}
