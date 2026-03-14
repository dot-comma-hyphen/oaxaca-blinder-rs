use polars::prelude::DataFrame;
use pyo3_polars::PyDataFrame;
fn test(df: PyDataFrame) -> DataFrame {
    df.0
}
