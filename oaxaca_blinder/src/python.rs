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
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
/// * `outcome` - Name of the outcome variable
/// * `predictors` - List of predictor variable names
/// * `categorical_predictors` - List of categorical predictor names
/// * `group` - Name of the group variable
/// * `reference_group` - Value representing the reference group
/// * `bootstrap_reps` - Number of bootstrap replications (default: 100)
#[pyfunction]
#[pyo3(signature = (csv_path, outcome, predictors, categorical_predictors, group, reference_group, bootstrap_reps=100))]
fn decompose_from_csv(
    csv_path: &str,
    outcome: &str,
    predictors: Vec<String>,
    categorical_predictors: Vec<String>,
    group: &str,
    reference_group: &str,
    bootstrap_reps: usize,
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

/// Python module for oaxaca_blinder
#[pymodule]
fn oaxaca_blinder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decompose_from_csv, m)?)?;
    m.add_class::<PyOaxacaResults>()?;
    Ok(())
}
