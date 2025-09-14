use polars::prelude::*;
use oaxaca_blinder::OaxacaBuilder;

fn create_sample_dataframe() -> DataFrame {
    df!(
        "wage" => &[10.0, 12.0, 11.0, 13.0, 15.0, 20.0, 22.0, 21.0, 23.0, 25.0],
        "education" => &[12.0, 16.0, 14.0, 16.0, 18.0, 12.0, 16.0, 14.0, 16.0, 18.0],
        "gender" => &["F", "F", "F", "F", "F", "M", "M", "M", "M", "M"]
    ).unwrap()
}

#[test]
fn test_full_run() {
    let df = create_sample_dataframe();

    let results = OaxacaBuilder::new(df, "wage", "gender", "F")
        .predictors(&["education"])
        .bootstrap_reps(50) // Use a small number of reps for a fast test
        .run()
        .expect("Oaxaca run failed");

    let male_wage = 22.2;
    let female_wage = 12.2;
    let expected_gap = male_wage - female_wage;

    // Check that the calculated gap is correct
    assert!((results.total_gap() - expected_gap).abs() < 1e-9);

    // Check that the two-fold decomposition sums to the total gap
    let explained = results.two_fold().aggregate().iter().find(|c| c.name() == "explained").unwrap().estimate();
    let unexplained = results.two_fold().aggregate().iter().find(|c| c.name() == "unexplained").unwrap().estimate();
    assert!((explained + unexplained - results.total_gap()).abs() < 1e-9);

    // Check that the number of observations is correct
    assert_eq!(*results.n_a(), 5);
    assert_eq!(*results.n_b(), 5);

    // Call summary to make sure it doesn't panic
    results.summary();
}
