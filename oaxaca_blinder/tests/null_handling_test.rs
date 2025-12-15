use oaxaca_blinder::OaxacaBuilder;
use polars::prelude::*;

#[test]
fn test_null_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Create data with nulls
    // A: 10, 12, 11 (3 obs kept)
    // B: 15, 16, 17 (3 obs kept)
    // Plus nulls to drop
    let s0 = Series::new(
        "outcome".into(),
        &[
            Some(10.0),
            Some(12.0),
            Some(11.0),
            None, // A
            Some(15.0),
            Some(16.0),
            Some(17.0),
            Some(18.0), // B
        ],
    );
    let s1 = Series::new("group".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
    let s2 = Series::new(
        "education".into(),
        &[
            Some(10.0),
            Some(12.0),
            Some(11.0),
            Some(12.0), // A: Var (10, 12, 11)
            Some(14.0),
            Some(16.0),
            Some(15.0),
            None, // B: Var (14, 16, 15)
        ],
    );

    // Rows:
    // 0: A, Out:10, Ed:12 -> Keep
    // 1: A, Out:12, Ed:12 -> Keep
    // 2: A, Out:11, Ed:12 -> Keep
    // 3: A, Out:Null, Ed:12 -> Drop
    // 4: B, Out:15, Ed:16 -> Keep
    // 5: B, Out:16, Ed:16 -> Keep
    // 6: B, Out:17, Ed:16 -> Keep
    // 7: B, Out:18, Ed:Null -> Drop

    let df = DataFrame::new(vec![s0.into(), s1.into(), s2.into()])?;

    let results = OaxacaBuilder::new(df, "outcome", "group", "B")
        .predictors(&["education"])
        .run()?;

    // Should have 3 obs in A and 3 obs in B
    assert_eq!(
        *results.n_a(),
        3,
        "Expected 3 observation in Group A after null dropping"
    );
    assert_eq!(
        *results.n_b(),
        3,
        "Expected 3 observation in Group B after null dropping"
    );

    Ok(())
}
