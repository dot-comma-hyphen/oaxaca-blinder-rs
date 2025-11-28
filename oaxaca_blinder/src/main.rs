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

    /// The type of analysis to perform [choices: mean, quantile, akm, match]
    #[arg(long, default_value = "mean")]
    analysis_type: String,

    /// Specifies the reference coefficients for the two-fold decomposition (for mean analysis) [choices: group_a, group_b, pooled, weighted]
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

    /// R-style formula for the model (e.g., "wage ~ education + experience + C(sector)")
    #[arg(long)]
    formula: Option<String>,

    /// Column name for sample weights (for WLS)
    #[arg(long)]
    weights: Option<String>,

    /// Outcome variable for the selection equation (Heckman correction)
    #[arg(long)]
    selection_outcome: Option<String>,

    /// Predictors for the selection equation (Heckman correction)
    #[arg(long, value_delimiter = ',')]
    selection_predictors: Option<Vec<String>>,

    /// Path to export results as JSON
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Path to export results as Markdown
    #[arg(long)]
    output_markdown: Option<PathBuf>,

    /// Worker ID column for AKM analysis (Abowd-Kramarz-Margolis model)
    #[arg(long)]
    worker_id: Option<String>,

    /// Firm ID column for AKM analysis (Abowd-Kramarz-Margolis model)
    #[arg(long)]
    firm_id: Option<String>,

    /// Number of neighbors for matching
    #[arg(long, default_value_t = 1)]
    k_neighbors: usize,

    /// Matching method [choices: euclidean, mahalanobis, psm]
    #[arg(long, default_value = "euclidean")]
    matching_method: String,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let df = CsvReader::from_path(&args.data)?
        .has_header(true)
        .finish()?;

    if args.analysis_type == "mean" {
        let reference_coeffs = match args.ref_coeffs.as_str() {
            "group_a" => ReferenceCoefficients::GroupA,
            "group_b" => ReferenceCoefficients::GroupB,
            "pooled" => ReferenceCoefficients::Pooled,
            "weighted" => ReferenceCoefficients::Weighted,
            _ => return Err("Invalid reference coefficient type".into()),
        };

        let mut builder = if let Some(formula) = &args.formula {
            OaxacaBuilder::from_formula(df, formula, &args.group, &args.reference)?
        } else {
            let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
            let categorical_predictors: Vec<&str> = args
                .categorical
                .as_ref()
                .map(|v| v.iter().map(AsRef::as_ref).collect())
                .unwrap_or_else(Vec::new);
            
            let mut b = OaxacaBuilder::new(df, &args.outcome, &args.group, &args.reference);
            b.predictors(&predictors)
             .categorical_predictors(&categorical_predictors);
            b
        };

        builder.bootstrap_reps(args.bootstrap_reps)
               .reference_coefficients(reference_coeffs);

        if let Some(weights) = &args.weights {
            builder.weights(weights);
        }

        if let Some(sel_outcome) = &args.selection_outcome {
            if let Some(sel_predictors) = &args.selection_predictors {
                let sel_preds_refs: Vec<&str> = sel_predictors.iter().map(AsRef::as_ref).collect();
                builder.heckman_selection(sel_outcome, &sel_preds_refs);
            } else {
                return Err("Selection predictors must be provided if selection outcome is specified".into());
            }
        }

        let results = builder.run()?;
        results.summary();
        
        if let Some(path) = args.output_json {
            let json = results.to_json().map_err(|e| format!("Failed to serialize to JSON: {}", e))?;
            std::fs::write(path, json)?;
        }

        if let Some(path) = args.output_markdown {
            let md = results.to_markdown();
            std::fs::write(path, md)?;
        }
        
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
    } else if args.analysis_type == "akm" {
        use oaxaca_blinder::AkmBuilder;
        
        let worker_col = args.worker_id.ok_or("Worker ID is required for AKM analysis")?;
        let firm_col = args.firm_id.ok_or("Firm ID is required for AKM analysis")?;
        
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        
        let builder = AkmBuilder::new(df, &args.outcome, &worker_col, &firm_col)
            .controls(&predictors);
            
        let results = builder.run().map_err(|e| format!("AKM estimation failed: {:?}", e))?;
        
        println!("AKM Estimation Results");
        println!("Method: Alternating Projections (MAP) on Largest Connected Set");
        println!("----------------------");
        println!("R-squared: {:.4}", results.r2);
        println!("Beta Coefficients:");
        for (i, name) in args.predictors.iter().enumerate() {
            if i < results.beta.len() {
                println!("  {}: {:.4}", name, results.beta[i]);
            }
        }
        
        // Export effects if requested?
        // For now just summary.
        // Export effects if requested?
        // For now just summary.
    } else if args.analysis_type == "match" {
        use oaxaca_blinder::MatchingEngine;
        
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        let engine = MatchingEngine::new(df, &args.group, &args.outcome, &predictors);
        
        let weights = if args.matching_method == "psm" {
            engine.match_psm(args.k_neighbors)
                .map_err(|e| format!("Matching failed: {:?}", e))?
        } else {
            let use_mahalanobis = args.matching_method == "mahalanobis";
            engine.run_matching(args.k_neighbors, use_mahalanobis)
                .map_err(|e| format!("Matching failed: {:?}", e))?
        };
        
        // Output weights
        // If JSON output is requested, save weights
        if let Some(path) = args.output_json {
            let json = serde_json::to_string(&weights)?;
            std::fs::write(path, json)?;
        } else {
            println!("Matching completed. Generated {} weights.", weights.len());
            println!("First 10 weights: {:?}", weights.iter().take(10).collect::<Vec<_>>());
        }
    } else {
        return Err(format!("Unknown analysis type: {}", args.analysis_type).into());
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
