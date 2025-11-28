use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass]
struct PyOaxacaResults {
    total_gap: f64,
    explained: f64,
    unexplained: f64,
    detailed_explained: HashMap<String, f64>,
    detailed_unexplained: HashMap<String, f64>,
}

#[pymethods]
impl PyOaxacaResults {
    #[getter]
    fn total_gap(&self) -> f64 {
        self.total_gap
    }

    #[getter]
    fn explained(&self) -> f64 {
        self.explained
    }

    #[getter]
    fn unexplained(&self) -> f64 {
        self.unexplained
    }

    #[getter]
    fn detailed_explained(&self) -> HashMap<String, f64> {
        self.detailed_explained.clone()
    }

    #[getter]
    fn detailed_unexplained(&self) -> HashMap<String, f64> {
        self.detailed_unexplained.clone()
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_gap", self.total_gap)?;
        dict.set_item("explained", self.explained)?;
        dict.set_item("unexplained", self.unexplained)?;
        dict.set_item("detailed_explained", self.detailed_explained.clone())?;
        dict.set_item("detailed_unexplained", self.detailed_unexplained.clone())?;
        Ok(dict.into())
    }
}

/// Performs Oaxaca-Blinder decomposition from a CSV file.
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, predictors, categorical_predictors, group, reference_group, bootstrap_reps=100, weights=None, selection_outcome=None, selection_predictors=None))]
fn decompose_from_csv(
    csv_path: &str,
    outcome: &str,
    predictors: Vec<String>,
    categorical_predictors: Vec<String>,
    group: &str,
    reference_group: &str,
    bootstrap_reps: usize,
    weights: Option<String>,
    selection_outcome: Option<String>,
    selection_predictors: Option<Vec<String>>,
) -> PyResult<PyOaxacaResults> {
    use polars::prelude::*;
    use crate::OaxacaBuilder;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let pred_refs: Vec<&str> = predictors.iter().map(|s| s.as_str()).collect();
    let cat_refs: Vec<&str> = categorical_predictors.iter().map(|s| s.as_str()).collect();

    let mut builder = OaxacaBuilder::new(df, outcome, group, reference_group);
    builder.predictors(&pred_refs)
           .categorical_predictors(&cat_refs)
           .bootstrap_reps(bootstrap_reps);

    if let Some(w) = &weights {
        builder.weights(w);
    }

    if let (Some(sel_out), Some(sel_preds)) = (selection_outcome, selection_predictors) {
        let sel_preds_refs: Vec<&str> = sel_preds.iter().map(|s| s.as_str()).collect();
        builder.heckman_selection(&sel_out, &sel_preds_refs);
    }

    let results = builder.run()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decomposition failed: {:?}", e)))?;

    let explained = results.two_fold().aggregate().iter()
        .find(|c| c.name() == "explained")
        .map(|c| *c.estimate())
        .unwrap_or(0.0);

    let unexplained = results.two_fold().aggregate().iter()
        .find(|c| c.name() == "unexplained")
        .map(|c| *c.estimate())
        .unwrap_or(0.0);

    let mut detailed_explained = HashMap::new();
    for comp in results.two_fold().detailed_explained() {
        detailed_explained.insert(comp.name().to_string(), *comp.estimate());
    }

    let mut detailed_unexplained = HashMap::new();
    for comp in results.two_fold().detailed_unexplained() {
        detailed_unexplained.insert(comp.name().to_string(), *comp.estimate());
    }

    Ok(PyOaxacaResults {
        total_gap: *results.total_gap(),
        explained,
        unexplained,
        detailed_explained,
        detailed_unexplained,
    })
}

/// Performs Quantile Decomposition (RIF-Regression) from a CSV file.
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, predictors, categorical_predictors, group, reference_group, quantile, bootstrap_reps=100, weights=None))]
fn decompose_quantile_from_csv(
    csv_path: &str,
    outcome: &str,
    predictors: Vec<String>,
    categorical_predictors: Vec<String>,
    group: &str,
    reference_group: &str,
    quantile: f64,
    bootstrap_reps: usize,
    weights: Option<String>,
) -> PyResult<PyOaxacaResults> {
    use polars::prelude::*;
    use crate::OaxacaBuilder;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let pred_refs: Vec<&str> = predictors.iter().map(|s| s.as_str()).collect();
    let cat_refs: Vec<&str> = categorical_predictors.iter().map(|s| s.as_str()).collect();

    let mut builder = OaxacaBuilder::new(df, outcome, group, reference_group);
    builder.predictors(&pred_refs)
           .categorical_predictors(&cat_refs)
           .bootstrap_reps(bootstrap_reps);

    if let Some(w) = &weights {
        builder.weights(w);
    }

    let results = builder.decompose_quantile(quantile)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Quantile decomposition failed: {:?}", e)))?;

    let explained = results.two_fold().aggregate().iter()
        .find(|c| c.name() == "explained")
        .map(|c| *c.estimate())
        .unwrap_or(0.0);

    let unexplained = results.two_fold().aggregate().iter()
        .find(|c| c.name() == "unexplained")
        .map(|c| *c.estimate())
        .unwrap_or(0.0);

    let mut detailed_explained = HashMap::new();
    for comp in results.two_fold().detailed_explained() {
        detailed_explained.insert(comp.name().to_string(), *comp.estimate());
    }

    let mut detailed_unexplained = HashMap::new();
    for comp in results.two_fold().detailed_unexplained() {
        detailed_unexplained.insert(comp.name().to_string(), *comp.estimate());
    }

    Ok(PyOaxacaResults {
        total_gap: *results.total_gap(),
        explained,
        unexplained,
        detailed_explained,
        detailed_unexplained,
    })
}

/// Optimizes budget to reduce pay gap.
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, predictors, categorical_predictors, group, reference_group, budget, target_gap))]
fn optimize_budget_from_csv(
    csv_path: &str,
    outcome: &str,
    predictors: Vec<String>,
    categorical_predictors: Vec<String>,
    group: &str,
    reference_group: &str,
    budget: f64,
    target_gap: f64,
) -> PyResult<Vec<HashMap<String, f64>>> {
    use polars::prelude::*;
    use crate::OaxacaBuilder;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let pred_refs: Vec<&str> = predictors.iter().map(|s| s.as_str()).collect();
    let cat_refs: Vec<&str> = categorical_predictors.iter().map(|s| s.as_str()).collect();

    let mut builder = OaxacaBuilder::new(df, outcome, group, reference_group);
    builder.predictors(&pred_refs)
           .categorical_predictors(&cat_refs)
           .bootstrap_reps(0); // No bootstrap needed for budget optimization

    let results = builder.run()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decomposition failed: {:?}", e)))?;

    let adjustments = results.optimize_budget(budget, target_gap);

    let mut py_adjustments = Vec::new();
    for adj in adjustments {
        let mut map = HashMap::new();
        map.insert("index".to_string(), adj.index as f64);
        map.insert("original_residual".to_string(), adj.original_residual);
        map.insert("adjustment".to_string(), adj.adjustment);
        py_adjustments.push(map);
    }

    Ok(py_adjustments)
}

/// Performs DFL Reweighting.
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, group, reference_group, predictors))]
fn run_dfl_from_csv(
    csv_path: &str,
    outcome: &str,
    group: &str,
    reference_group: &str,
    predictors: Vec<String>,
) -> PyResult<HashMap<String, Vec<f64>>> {
    use polars::prelude::*;
    use crate::run_dfl;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let result = run_dfl(&df, outcome, group, reference_group, &predictors)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("DFL failed: {:?}", e)))?;

    let mut map = HashMap::new();
    map.insert("grid".to_string(), result.grid);
    map.insert("density_a".to_string(), result.density_a);
    map.insert("density_b".to_string(), result.density_b);
    map.insert("density_b_counterfactual".to_string(), result.density_b_counterfactual);

    Ok(map)
}

/// Performs matching using the Matching Engine.
#[pyfunction]
#[pyo3(signature = (csv_path, treatment, outcome, covariates, k=1, method="euclidean"))]
fn match_units(
    csv_path: &str,
    treatment: &str,
    outcome: &str,
    covariates: Vec<String>,
    k: usize,
    method: &str,
) -> PyResult<Vec<f64>> {
    use polars::prelude::*;
    use crate::MatchingEngine;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
    let engine = MatchingEngine::new(df, treatment, outcome, &cov_refs);

    let weights = if method == "psm" {
        engine.match_psm(k)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Matching failed: {:?}", e)))?
    } else {
        let use_mahalanobis = method == "mahalanobis";
        engine.run_matching(k, use_mahalanobis)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Matching failed: {:?}", e)))?
    };

    Ok(weights)
}

/// Python module for oaxaca_blinder
#[pymodule]
fn oaxaca_blinder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decompose_from_csv, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_quantile_from_csv, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_budget_from_csv, m)?)?;
    m.add_function(wrap_pyfunction!(run_dfl_from_csv, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_akm, m)?)?;
    m.add_function(wrap_pyfunction!(match_units, m)?)?;
    m.add_class::<PyOaxacaResults>()?;
    m.add_class::<PyAkmResult>()?;
    Ok(())
}

#[pyclass]
struct PyAkmResult {
    beta: Vec<f64>,
    worker_effects: HashMap<String, f64>,
    firm_effects: HashMap<String, f64>,
    r2: f64,
}

#[pymethods]
impl PyAkmResult {
    #[getter]
    fn beta(&self) -> Vec<f64> {
        self.beta.clone()
    }

    #[getter]
    fn worker_effects(&self) -> HashMap<String, f64> {
        self.worker_effects.clone()
    }

    #[getter]
    fn firm_effects(&self) -> HashMap<String, f64> {
        self.firm_effects.clone()
    }

    #[getter]
    fn r2(&self) -> f64 {
        self.r2
    }
}

/// Estimates the AKM model from a CSV file.
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, worker_col, firm_col, controls=None, tolerance=1e-8, max_iters=1000))]
fn estimate_akm(
    csv_path: &str,
    outcome: &str,
    worker_col: &str,
    firm_col: &str,
    controls: Option<Vec<String>>,
    tolerance: f64,
    max_iters: usize,
) -> PyResult<PyAkmResult> {
    use polars::prelude::*;
    use crate::AkmBuilder;

    let df = LazyCsvReader::new(csv_path)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read CSV: {}", e)))?
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse CSV: {}", e)))?;

    let mut builder = AkmBuilder::new(df, outcome, worker_col, firm_col)
        .tolerance(tolerance)
        .max_iters(max_iters);
        
    if let Some(c) = controls {
        let refs: Vec<&str> = c.iter().map(|s| s.as_str()).collect();
        builder = builder.controls(&refs);
    }

    let result = builder.run()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("AKM estimation failed: {:?}", e)))?;

    // Convert DataFrames to HashMaps for Python
    let mut worker_effects = HashMap::new();
    let w_ids = result.worker_effects.column(worker_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        .cast(&DataType::String)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let w_effs = result.worker_effects.column("effect")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        .f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let w_ids_utf8 = w_ids.str()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    for (id_opt, eff_opt) in w_ids_utf8.into_iter().zip(w_effs.into_iter()) {
        if let (Some(id), Some(eff)) = (id_opt, eff_opt) {
            worker_effects.insert(id.to_string(), eff);
        }
    }

    let mut firm_effects = HashMap::new();
    let f_ids = result.firm_effects.column(firm_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        .cast(&DataType::String)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let f_effs = result.firm_effects.column("effect")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        .f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let f_ids_utf8 = f_ids.str()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    for (id_opt, eff_opt) in f_ids_utf8.into_iter().zip(f_effs.into_iter()) {
        if let (Some(id), Some(eff)) = (id_opt, eff_opt) {
            firm_effects.insert(id.to_string(), eff);
        }
    }

    Ok(PyAkmResult {
        beta: result.beta.iter().cloned().collect(),
        worker_effects,
        firm_effects,
        r2: result.r2,
    })
}
