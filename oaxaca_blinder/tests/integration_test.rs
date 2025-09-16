use polars::prelude::*;
use oaxaca_blinder::{OaxacaBuilder, ReferenceCoefficients, QuantileDecompositionBuilder};

fn create_sample_dataframe() -> DataFrame {
    df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap()
}

// Helper function to avoid test duplication
fn run_and_check(builder: OaxacaBuilder, expected_gap: f64) {
    let results = builder.run().expect("Oaxaca run failed");

    // Check that the calculated gap is correct
    assert!((results.total_gap() - expected_gap).abs() < 1e-9);

    // Check that the two-fold decomposition sums to the total gap
    let explained = results.two_fold().aggregate().iter().find(|c| c.name() == "explained").unwrap().estimate();
    let unexplained = results.two_fold().aggregate().iter().find(|c| c.name() == "unexplained").unwrap().estimate();
    let total_gap = results.total_gap();
    println!("Explained: {}, Unexplained: {}, Sum: {}, Total Gap: {}", explained, unexplained, explained + unexplained, total_gap);
    assert!((explained + unexplained - results.total_gap()).abs() < 1e-9, "Decomposition does not sum to total gap");

    // Check that the number of observations is correct
    assert_eq!(*results.n_a(), 5);
    assert_eq!(*results.n_b(), 5);

    // Call summary to make sure it doesn't panic
    results.summary();
}

#[test]
fn test_full_run_group_b_ref() {
    let df = create_sample_dataframe();
    let builder = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(5); // Default is GroupB
    run_and_check(builder, 10.0);
}

#[test]
fn test_full_run_group_a_ref() {
    let df = create_sample_dataframe();
    let builder = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(5)
        .reference_coefficients(ReferenceCoefficients::GroupA);
    run_and_check(builder, 10.0);
}

#[test]
fn test_full_run_pooled_ref() {
    let df = create_sample_dataframe();
    let builder = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(5)
        .reference_coefficients(ReferenceCoefficients::Pooled);
    run_and_check(builder, 10.0);
}

#[test]
fn test_full_run_weighted_ref() {
    let df = create_sample_dataframe();
    let builder = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(5)
        .reference_coefficients(ReferenceCoefficients::Weighted);
    run_and_check(builder, 10.0);
}

#[test]
fn test_with_categorical_variable() {
    let df = df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"],
        "union" => &["none", "union", "union_plus", "none", "union", "union_plus", "none", "union", "union_plus", "none"]
    ).unwrap();

    let builder = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .categorical_predictors(&["union"])
        .normalize(&["union"])
        .bootstrap_reps(5);

    run_and_check(builder, 10.0);
}

#[test]
fn test_quantile_decomposition() {
    let df = df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0, 9.0, 18.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0, 10.0, 20.0],
        "gender" => &["F", "F", "F", "F", "F", "F", "M", "M", "M", "M", "M", "M"]
    )
    .unwrap();

    let quantiles_to_test = &[0.25, 0.5, 0.75];
    let results = QuantileDecompositionBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .quantiles(quantiles_to_test)
        .simulations(50) // Low number for fast testing
        .bootstrap_reps(20) // Low number for fast testing
        .run()
        .unwrap();

    assert!(results.results_by_quantile().contains_key("q25"));
    assert!(results.results_by_quantile().contains_key("q50"));
    assert!(results.results_by_quantile().contains_key("q75"));

    for key in &["q25", "q50", "q75"] {
        let detail = results.results_by_quantile().get(*key).unwrap();
        let gap = detail.total_gap().estimate();
        let chars = detail.characteristics_effect().estimate();
        let coeffs = detail.coefficients_effect().estimate();
        assert!((chars + coeffs - gap).abs() < 1e-9);
    }

    results.summary();
}
