import os

# Goal 1
with open("oaxaca_blinder/src/akm.rs", "r") as f:
    akm_code = f.read()

akm_code = akm_code.replace(
    ".map(|opt| worker_map.get(opt.unwrap()).unwrap().clone())",
    ".map(|opt| opt.ok_or_else(|| AkmError::NotEnoughData(\"Null worker ID encountered\".to_string()))).collect::<Result<Vec<_>, _>>()?\n        .into_iter()\n        .map(|id| worker_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Worker ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?"
)

akm_code = akm_code.replace(
    ".map(|opt| firm_map.get(opt.unwrap()).unwrap().clone())",
    ".map(|opt| opt.ok_or_else(|| AkmError::NotEnoughData(\"Null firm ID encountered\".to_string()))).collect::<Result<Vec<_>, _>>()?\n        .into_iter()\n        .map(|id| firm_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Firm ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?"
)

# Goal 2 (AKM part)
akm_code = akm_code.replace(
    "let y: Vec<f64> = y_series.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();",
    "let y: Vec<f64> = y_series.into_iter().map(|opt| opt.ok_or_else(|| AkmError::NotEnoughData(\"Null outcome encountered\".to_string()))).collect::<Result<Vec<_>, _>>()?;"
)

# Goal 5 (AKM part)
old_akm_norm = """
        // Normalize one firm to zero (e.g., first firm) to identify the model
        // Or mean center? Usually one reference is dropped.
        // Let's center firm effects to mean zero and absorb the constant into alpha?
        // Or just set psi[0] = 0.
        // The prompt says: "Note that one reference firm (or worker) must be normalized to zero"
        let ref_val = psi[0];
        for j in 0..n_firms {
            psi[j] -= ref_val;
        }
        // Add ref_val to alpha to keep prediction same?
        // Y = alpha + psi. If psi -> psi - c, then alpha -> alpha + c.
        for i in 0..n_workers {
            alpha[i] += ref_val;
        }

        // Check convergence
"""
new_akm_norm = """
        // Check convergence
"""
if old_akm_norm in akm_code:
    akm_code = akm_code.replace(old_akm_norm, new_akm_norm)
else:
    print("WARNING: Could not find old_akm_norm in akm.rs")

after_loop_norm = """
    if iter >= max_iters {
        return Err(AkmError::ConvergenceFailed(format!(
            "recover_fe failed to converge within {} iterations",
            max_iters
        )));
    }

    // Normalize one firm to zero (e.g., first firm) to identify the model
    let ref_val = psi[0];
    for j in 0..n_firms {
        psi[j] -= ref_val;
    }
    for i in 0..n_workers {
        alpha[i] += ref_val;
    }

    Ok((alpha, psi))
"""
old_ok = """
    if iter >= max_iters {
        return Err(AkmError::ConvergenceFailed(format!(
            "recover_fe failed to converge within {} iterations",
            max_iters
        )));
    }

    Ok((alpha, psi))
"""
if old_ok.strip() in akm_code:
    akm_code = akm_code.replace(old_ok.strip(), after_loop_norm.strip())

with open("oaxaca_blinder/src/akm.rs", "w") as f:
    f.write(akm_code)

# Goal 1 (lib.rs)
with open("oaxaca_blinder/src/lib.rs", "r") as f:
    lib_code = f.read()

lib_code = lib_code.replace(
    'let y_sel_vec: Vec<f64> = y_sel_series\n            .into_iter()\n            .map(|opt| opt.expect("Selection outcome should be clean"))\n            .collect();\n        let y_sel = DVector::from_vec(y_sel_vec);',
    'let y_sel_vec: Result<Vec<f64>, OaxacaError> = y_sel_series\n            .into_iter()\n            .map(|opt| opt.ok_or_else(|| OaxacaError::InvalidGroupVariable("Selection outcome contains nulls".to_string())))\n            .collect();\n        let y_sel = DVector::from_vec(y_sel_vec?);'
)

# Goal 5 (lib.rs part)
old_lib_boot = """
        let group_a_name_owned = group_a_name.to_string();
        let group_b_name_owned = group_b_name.to_string();

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                // Stratified sampling: Sample from Group A and Group B separately
                let df_a = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_a_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;
                let df_b = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_b_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;
"""

new_lib_boot = """
        let group_a_name_owned = group_a_name.to_string();
        let group_b_name_owned = group_b_name.to_string();

        let df_a_global = df
            .filter(
                &df.column(&self.group)
                    .unwrap()
                    .as_materialized_series()
                    .equal(group_a_name_owned.as_str())
                    .unwrap(),
            )
            .unwrap();
        let df_b_global = df
            .filter(
                &df.column(&self.group)
                    .unwrap()
                    .as_materialized_series()
                    .equal(group_b_name_owned.as_str())
                    .unwrap(),
            )
            .unwrap();

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                let df_a = df_a_global.clone();
                let df_b = df_b_global.clone();
"""
if old_lib_boot in lib_code:
    lib_code = lib_code.replace(old_lib_boot, new_lib_boot)
else:
    print("WARNING: Could not find old_lib_boot in lib.rs")

with open("oaxaca_blinder/src/lib.rs", "w") as f:
    f.write(lib_code)

# Goal 1 (heckman.rs) and Goal 3 (Heckman part)
with open("oaxaca_blinder/src/heckman.rs", "r") as f:
    heckman_code = f.read()

heckman_code = heckman_code.replace(
    'let normal = Normal::new(0.0, 1.0).unwrap();',
    'let normal = Normal::new(0.0, 1.0).map_err(|e| OaxacaError::NalgebraError(format!("Failed to create normal distribution: {}", e)))?;'
)

old_heckman_imr = """
        let big_phi = normal.cdf(zg);
        if big_phi > 0.0 {
            phi / big_phi
        } else {
            0.0
        }
"""
new_heckman_imr = """
        let big_phi = normal.cdf(zg);
        (phi / big_phi.max(1e-300)).min(40.0)
"""
heckman_code = heckman_code.replace(old_heckman_imr, new_heckman_imr)

with open("oaxaca_blinder/src/heckman.rs", "w") as f:
    f.write(heckman_code)

# Goal 2 (quantile_decomposition.rs)
with open("oaxaca_blinder/src/quantile_decomposition.rs", "r") as f:
    quant_code = f.read()

quant_code = quant_code.replace(
    'let y_vec: Vec<f64> = y_series.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();',
    'let y_vec: Vec<f64> = y_series.into_iter().map(|opt| opt.ok_or_else(|| OaxacaError::InvalidGroupVariable("Null outcome encountered".to_string()))).collect::<Result<Vec<_>, _>>()?;'
)
quant_code = quant_code.replace(
    'data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());',
    'data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));'
)

# Goal 5 (quantile_decomposition.rs)
old_quant_boot = """
        let group_a_name_owned = group_a_name.to_string();
        let group_b_name_owned = group_b_name.to_string();

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                // Stratified sampling: Sample from Group A and Group B separately
                let df_a = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_a_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;
                let df_b = df
                    .filter(
                        &df.column(&self.group)
                            .ok()?
                            .as_materialized_series()
                            .equal(group_b_name_owned.as_str())
                            .ok()?,
                    )
                    .ok()?;
"""

new_quant_boot = """
        let group_a_name_owned = group_a_name.to_string();
        let group_b_name_owned = group_b_name.to_string();

        let df_a_global = df
            .filter(
                &df.column(&self.group)
                    .unwrap()
                    .as_materialized_series()
                    .equal(group_a_name_owned.as_str())
                    .unwrap(),
            )
            .unwrap();
        let df_b_global = df
            .filter(
                &df.column(&self.group)
                    .unwrap()
                    .as_materialized_series()
                    .equal(group_b_name_owned.as_str())
                    .unwrap(),
            )
            .unwrap();

        let bootstrap_results: Vec<SinglePassResult> = (0..self.bootstrap_reps)
            .into_par_iter()
            .filter_map(|_| {
                let df_a = df_a_global.clone();
                let df_b = df_b_global.clone();
"""

if old_quant_boot in quant_code:
    quant_code = quant_code.replace(old_quant_boot, new_quant_boot)
else:
    print("WARNING: Could not find old_quant_boot in quantile_decomposition.rs")

with open("oaxaca_blinder/src/quantile_decomposition.rs", "w") as f:
    f.write(quant_code)


# Goal 2 (inference.rs, math/kde.rs, math/rif.rs, matching/engine.rs, dfl.rs)
with open("oaxaca_blinder/src/inference.rs", "r") as f:
    inf_code = f.read()
inf_code = inf_code.replace('sorted_estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());', 'sorted_estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));')
with open("oaxaca_blinder/src/inference.rs", "w") as f:
    f.write(inf_code)

with open("oaxaca_blinder/src/math/kde.rs", "r") as f:
    kde_code = f.read()
kde_code = kde_code.replace('sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());', 'sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));')
with open("oaxaca_blinder/src/math/kde.rs", "w") as f:
    f.write(kde_code)

with open("oaxaca_blinder/src/math/rif.rs", "r") as f:
    rif_code = f.read()
rif_code = rif_code.replace('sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());', 'sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));')

# Goal 3 (RIF part)
old_rif_quantile = """
    // 1. Calculate Sample Quantile (Q_tau)
    let mut sorted_y = y_vec.clone();
    sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_index = (quantile * n).ceil() as usize;
    let q_index = if q_index == 0 { 0 } else { q_index - 1 };
    let q_tau = sorted_y[q_index.min(sorted_y.len() - 1)];
"""

new_rif_quantile = """
    // 1. Calculate Sample Quantile (Q_tau) using R Type 7 interpolation
    let mut sorted_y = y_vec.clone();
    sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let h = (n - 1.0) * quantile;
    let h_floor = h.floor();
    let h_ceil = h.ceil();
    let frac = h - h_floor;
    let q_tau = if h_floor == h_ceil {
        sorted_y[h_floor as usize]
    } else {
        let y0 = sorted_y[h_floor as usize];
        let y1 = sorted_y[h_ceil as usize];
        y0 + frac * (y1 - y0)
    };
"""
rif_code = rif_code.replace(old_rif_quantile.strip(), new_rif_quantile.strip())
with open("oaxaca_blinder/src/math/rif.rs", "w") as f:
    f.write(rif_code)

with open("oaxaca_blinder/src/matching/engine.rs", "r") as f:
    engine_code = f.read()
engine_code = engine_code.replace('let v: Vec<f64> = s.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();', 'let v: Vec<f64> = s.into_iter().map(|opt| opt.ok_or_else(|| OaxacaError::InvalidGroupVariable("Null values in outcomes".to_string()))).collect::<Result<Vec<_>, _>>()?;')
with open("oaxaca_blinder/src/matching/engine.rs", "w") as f:
    f.write(engine_code)

with open("oaxaca_blinder/src/dfl.rs", "r") as f:
    dfl_code = f.read()
dfl_code = dfl_code.replace('let val = outcome_series.get(i).unwrap_or(0.0);', 'let val = outcome_series.get(i).ok_or_else(|| OaxacaError::InvalidGroupVariable("Null outcome encountered in DFL".to_string()))?;')
with open("oaxaca_blinder/src/dfl.rs", "w") as f:
    f.write(dfl_code)

# Goal 3 (Logit)
with open("oaxaca_blinder/src/math/logit.rs", "r") as f:
    logit_code = f.read()
logit_code = logit_code.replace("let probs: DVector<f64> = xb.map(sigmoid);", "let probs: DVector<f64> = xb.map(|z| sigmoid(z).clamp(1e-10, 1.0 - 1e-10));")
logit_code = logit_code.replace("let probs = xb.map(sigmoid);", "let probs = xb.map(|z| sigmoid(z).clamp(1e-10, 1.0 - 1e-10));")
with open("oaxaca_blinder/src/math/logit.rs", "w") as f:
    f.write(logit_code)

# Goal 4 (OLS)
with open("oaxaca_blinder/src/math/ols.rs", "r") as f:
    ols_code = f.read()

old_wls = """
        let w_sqrt = w.map(|v| v.sqrt());

        // Scale X by sqrt(weights) row-wise
        let mut x_w = x.clone();
        for j in 0..x.ncols() {
            let mut col = x_w.column_mut(j);
            col.component_mul_assign(&w_sqrt);
        }

        // Scale y by sqrt(weights)
        let y_w = y.component_mul(&w_sqrt);

        let xtx = x_w.transpose() * &x_w;
        let xty = x_w.transpose() * &y_w;

        // Effective sample size? Usually just sum of weights or N?
        // For variance estimation in survey data, it's complicated.
        // But for standard WLS (heteroskedasticity), we use N.
        // If weights are frequency weights, we use sum(w).
        // Let's assume sampling weights/frequency weights -> sum(w).
        let n = w.sum();
"""

new_wls = """
        for weight in w.iter() {
            if *weight < 0.0 {
                return Err(OaxacaError::InvalidGroupVariable("Weights cannot be negative".to_string()));
            }
        }

        let w_sqrt = w.map(|v| v.sqrt());

        // Scale X by sqrt(weights) row-wise
        let mut x_w = x.clone();
        for j in 0..x.ncols() {
            let mut col = x_w.column_mut(j);
            col.component_mul_assign(&w_sqrt);
        }

        // Scale y by sqrt(weights)
        let y_w = y.component_mul(&w_sqrt);

        let xtx = x_w.transpose() * &x_w;
        let xty = x_w.transpose() * &y_w;

        let n = x.nrows() as f64;
"""

ols_code = ols_code.replace(old_wls.strip(), new_wls.strip())

with open("oaxaca_blinder/src/math/ols.rs", "w") as f:
    f.write(ols_code)
