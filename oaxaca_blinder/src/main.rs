use clap::{CommandFactory, Parser};
use oaxaca_blinder::{OaxacaBuilder, QuantileDecompositionBuilder, ReferenceCoefficients};
use polars::prelude::*;
use std::error::Error;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input CSV data file
    #[arg(short, long)]
    data: PathBuf,

    /// The name of the column that contains the outcome variable
    #[arg(long)]
    outcome: String,

    /// The name of the column that divides the data into two distinct groups
    #[arg(long)]
    group: String,

    /// The specific value within the group column that identifies the reference group
    #[arg(long)]
    reference: String,

    /// A comma-separated string of column names to be used as numerical predictor variables
    #[arg(long, value_delimiter = ',')]
    predictors: Vec<String>,

    /// A comma-separated string of column names to be treated as categorical variables
    #[arg(long, value_delimiter = ',')]
    categorical: Option<Vec<String>>,

    /// The type of analysis to perform
    #[arg(long, default_value = "mean")]
    analysis_type: String,

    /// Specifies the reference coefficients for the two-fold decomposition (for mean analysis)
    #[arg(long, default_value = "group_b")]
    ref_coeffs: String,

    /// A comma-separated string of quantiles to analyze (for quantile analysis)
    #[arg(long, value_delimiter = ',')]
    quantiles: Option<Vec<f64>>,

    /// The number of bootstrap replications for calculating standard errors
    #[arg(long, default_value_t = 500)]
    bootstrap_reps: usize,

    /// The number of simulations for the Machado-Mata algorithm (for quantile analysis)
    #[arg(long, default_value_t = 1000)]
    simulations: usize,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let df = CsvReader::from_path(args.data)?
        .has_header(true)
        .finish()?;

    if args.analysis_type == "mean" {
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        let categorical_predictors: Vec<&str> = args
            .categorical
            .as_ref()
            .map(|v| v.iter().map(AsRef::as_ref).collect())
            .unwrap_or_else(Vec::new);

        let reference_coeffs = match args.ref_coeffs.as_str() {
            "group_a" => ReferenceCoefficients::GroupA,
            "group_b" => ReferenceCoefficients::GroupB,
            "pooled" => ReferenceCoefficients::Pooled,
            "weighted" => ReferenceCoefficients::Weighted,
            _ => return Err("Invalid reference coefficient type".into()),
        };

        let mut builder = OaxacaBuilder::new(df, &args.outcome, &args.group, &args.reference);

        builder
            .predictors(&predictors)
            .categorical_predictors(&categorical_predictors)
            .bootstrap_reps(args.bootstrap_reps)
            .reference_coefficients(reference_coeffs);

        let results = builder.run()?;
        results.summary();
    } else if args.analysis_type == "quantile" {
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        let quantiles = args
            .quantiles
            .unwrap_or_else(|| vec![0.1, 0.25, 0.5, 0.75, 0.9]);

        let categorical_predictors: Vec<&str> = args
            .categorical
            .as_ref()
            .map(|v| v.iter().map(AsRef::as_ref).collect())
            .unwrap_or_else(Vec::new);

        let mut builder = QuantileDecompositionBuilder::new(df, &args.outcome, &args.group, &args.reference);

        builder
            .predictors(&predictors)
            .categorical_predictors(&categorical_predictors)
            .quantiles(&quantiles)
            .bootstrap_reps(args.bootstrap_reps)
            .simulations(args.simulations);

        let results = builder.run()?;

        results.summary();
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        let _ = Args::command().print_help();
        std::process::exit(1);
    }
}
