use polars::prelude::*;
use oaxaca_blinder::OaxacaBuilder;

#[test]
fn test_formula_interface() -> Result<(), Box<dyn std::error::Error>> {
    let df = df!(
        "wage" => &[10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 11.0, 13.0, 16.0, 19.0, 21.0, 23.0],
        "education" => &[12.0, 14.0, 16.0, 12.0, 14.0, 16.0, 13.0, 15.0, 17.0, 13.0, 15.0, 17.0],
        "experience" => &[5.0, 10.0, 15.0, 6.0, 11.0, 16.0, 7.0, 12.0, 17.0, 8.0, 13.0, 18.0],
        "gender" => &["F", "F", "F", "M", "M", "M", "F", "F", "F", "M", "M", "M"],
        "sector" => &["A", "B", "A", "B", "A", "B", "B", "A", "B", "A", "B", "A"]
    )?;

    let mut builder = OaxacaBuilder::from_formula(df, "wage ~ education + experience + C(sector)", "gender", "F")?;
    
    let res = builder.bootstrap_reps(0).run()?;
    
    println!("Formula interface test passed!");
    println!("Total gap: {:.4}", res.total_gap());
    
    Ok(())
}
