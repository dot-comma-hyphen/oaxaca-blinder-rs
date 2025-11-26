use polars::prelude::*;
use rand::prelude::*;
use std::fs::File;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_rows = 100_000;
    let mut rng = StdRng::seed_from_u64(42);

    // Predictors
    let mut education: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(8.0..20.0)).collect();
    let mut experience: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(0.0..40.0)).collect();
    let mut age: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(18.0..65.0)).collect();
    let mut tenure: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(0.0..20.0)).collect();
    let mut ability: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(0.0..100.0)).collect();
    let mut training: Vec<f64> = (0..n_rows).map(|_| if rng.gen_bool(0.3) { 1.0 } else { 0.0 }).collect();
    let mut performance: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(1.0..5.0)).collect();
    let mut hours: Vec<f64> = (0..n_rows).map(|_| rng.gen_range(20.0..60.0)).collect();
    let mut metro: Vec<f64> = (0..n_rows).map(|_| if rng.gen_bool(0.6) { 1.0 } else { 0.0 }).collect();
    let mut married: Vec<f64> = (0..n_rows).map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 }).collect();

    // Group (Gender: M/F)
    // Let's make F have slightly better education but lower returns
    let gender: Vec<String> = (0..n_rows).map(|i| {
        if rng.gen_bool(0.5) { 
            "M".to_string() 
        } else { 
            education[i] += 0.5; // F has slightly more education on average
            "F".to_string() 
        }
    }).collect();

    // Outcome (Wage)
    // Wage = Base + B1*Edu + B2*Exp + ... + Noise
    // Add discrimination: Lower intercept for F, maybe lower return on experience
    let wage: Vec<f64> = (0..n_rows).map(|i| {
        let is_female = if gender[i] == "F" { 1.0 } else { 0.0 };
        
        let mut w = 10.0; // Base
        w += 1.5 * education[i];
        w += 0.5 * experience[i];
        w += 0.1 * age[i];
        w += 0.3 * tenure[i];
        w += 0.05 * ability[i];
        w += 2.0 * training[i];
        w += 1.0 * performance[i];
        w += 0.2 * hours[i];
        w += 3.0 * metro[i];
        w += 1.5 * married[i];
        
        // Discrimination / Group Differences
        w -= 2.0 * is_female; // Direct penalty
        w -= 0.1 * experience[i] * is_female; // Lower return on experience for F
        
        // Box-Muller transform for normal distribution
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        w += z * 5.0; // Random error
        
        w.max(5.0) // Minimum wage
    }).collect();

    let mut df = df!(
        "wage" => wage,
        "gender" => gender,
        "education" => education,
        "experience" => experience,
        "age" => age,
        "tenure" => tenure,
        "ability" => ability,
        "training" => training,
        "performance" => performance,
        "hours" => hours,
        "metro" => metro,
        "married" => married
    )?;

    let file = File::create("benchmark_100k.csv")?;
    let mut writer = BufWriter::new(file);
    CsvWriter::new(&mut writer).finish(&mut df)?;

    println!("Generated benchmark_100k.csv with {} rows.", n_rows);
    Ok(())
}
