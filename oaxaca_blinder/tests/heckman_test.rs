use polars::prelude::*;
use oaxaca_blinder::OaxacaBuilder;
use rand::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::Normal;

#[test]
fn test_heckman_correction() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with selection bias
    // Z ~ N(0, 1)
    // u ~ N(0, 1), e ~ N(0, 1), corr(u, e) = 0.8
    // Selection: S = 1 if 0.5 * Z + u > 0
    // Outcome: Y = 1.0 + 2.0 * X + e (observed if S=1)
    // X is correlated with Z? Let's say X = Z + noise.
    
    let n = 2000;
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut z_vals = Vec::new();
    let mut x_vals = Vec::new();
    let mut s_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut group_vals = Vec::new();
    
    for _ in 0..n {
        let z: f64 = normal.sample(&mut rng);
        let x: f64 = z + 0.5 * normal.sample(&mut rng);
        
        let u: f64 = normal.sample(&mut rng);
        let e_uncorr: f64 = normal.sample(&mut rng);
        let rho = 0.8;
        let e = rho * u + (1.0 - rho*rho).sqrt() * e_uncorr;
        
        let s_latent = 0.5 * z + u;
        let s = if s_latent > 0.0 { 1.0 } else { 0.0 };
        
        let y = 1.0 + 2.0 * x + e;
        
        // Randomly assign group
        let group = if rng.gen_bool(0.5) { "A" } else { "B" };
        
        z_vals.push(z);
        x_vals.push(x);
        s_vals.push(s);
        y_vals.push(if s == 1.0 { Some(y) } else { None });
        group_vals.push(group);
    }
    
    let df = df!(
        "outcome" => y_vals,
        "x" => x_vals,
        "z" => z_vals,
        "selection" => s_vals,
        "group" => group_vals
    )?;
    
    // Run Oaxaca with Heckman
    let res = OaxacaBuilder::new(df, "outcome", "group", "B")
        .predictors(&["x"])
        .heckman_selection("selection", &["z"]) // Z is exclusion restriction
        .bootstrap_reps(0)
        .run()?;
        
    // Check that we have IMR in the results
    let explained = res.two_fold().detailed_explained();
    let has_imr = explained.iter().any(|c| c.name() == "IMR");
    assert!(has_imr, "IMR should be in detailed decomposition");
    
    println!("Heckman Decomposition Summary:");
    res.summary();
    
    Ok(())
}
