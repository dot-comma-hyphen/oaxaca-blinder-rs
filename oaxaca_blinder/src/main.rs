use clap::{CommandFactory, Parser, Subcommand};
use oaxaca_blinder::{OaxacaBuilder, QuantileDecompositionBuilder, ReferenceCoefficients};
use polars::prelude::*;
use serde::Serialize;
use std::error::Error;
use std::path::PathBuf;

#[derive(Serialize)]
struct OverallRow {
    name: String,
    coefficient: f64,
    std_error: f64,
    t_stat: f64,
    p_value: f64,
}

#[derive(Serialize)]
struct DetailedRow {
    variable: String,
    explained: f64,
    unexplained: f64,
}

struct ReportData {
    total_gap: String,
    explained_gap: String,
    unexplained_gap: String,
    overall: Vec<OverallRow>,
    detailed: Vec<DetailedRow>,
}

impl ReportData {
    fn render(&self) -> Result<String, Box<dyn Error>> {
        let mut overall_rows = String::new();
        for row in &self.overall {
            overall_rows.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>",
                row.name, row.coefficient, row.std_error, row.t_stat, row.p_value
            ));
        }

        let mut detailed_rows = String::new();
        for row in &self.detailed {
            detailed_rows.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>",
                row.variable, row.explained, row.unexplained, row.explained + row.unexplained
            ));
        }

        let template = include_str!("../templates/report.html");
        let html = template
            .replace("{{ total_gap }}", &self.total_gap)
            .replace("{{ explained_gap }}", &self.explained_gap)
            .replace("{{ unexplained_gap }}", &self.unexplained_gap)
            .replace("<!-- overall_rows -->", &overall_rows)
            .replace("<!-- detailed_rows -->", &detailed_rows);

        Ok(html)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    run_args: RunArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run an analysis and print results to the console or export to JSON/Markdown
    #[clap(name = "run")]
    Run(RunArgs),
    /// Generate a static HTML report from an analysis
    Report(ReportArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
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

#[derive(Parser, Debug)]
struct ReportArgs {
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

    /// Path to save the output HTML file
    #[arg(short, long)]
    output: PathBuf,
}

fn run_analysis(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let df = LazyCsvReader::new(&args.data)
        .with_has_header(true)
        .finish()?
        .collect()?;
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
        builder
            .bootstrap_reps(args.bootstrap_reps)
            .reference_coefficients(reference_coeffs);
        if let Some(weights) = &args.weights {
            builder.weights(weights);
        }
        if let Some(sel_outcome) = &args.selection_outcome {
            if let Some(sel_predictors) = &args.selection_predictors {
                let sel_preds_refs: Vec<&str> =
                    sel_predictors.iter().map(AsRef::as_ref).collect();
                builder.heckman_selection(sel_outcome, &sel_preds_refs);
            } else {
                return Err(
                    "Selection predictors must be provided if selection outcome is specified".into(),
                );
            }
        }
        let results = builder.run()?;
        results.summary();
        if let Some(path) = args.output_json {
            let json = results
                .to_json()
                .map_err(|e| format!("Failed to serialize to JSON: {}", e))?;
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
        let mut builder =
            QuantileDecompositionBuilder::new(df, &args.outcome, &args.group, &args.reference);
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
        let worker_col = args
            .worker_id
            .ok_or("Worker ID is required for AKM analysis")?;
        let firm_col = args
            .firm_id
            .ok_or("Firm ID is required for AKM analysis")?;
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        let builder =
            AkmBuilder::new(df, &args.outcome, &worker_col, &firm_col).controls(&predictors);
        let results = builder
            .run()
            .map_err(|e| format!("AKM estimation failed: {:?}", e))?;
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
    } else if args.analysis_type == "match" {
        use oaxaca_blinder::MatchingEngine;
        let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
        let engine = MatchingEngine::new(df, &args.group, &args.outcome, &predictors);
        let weights = if args.matching_method == "psm" {
            engine
                .match_psm(args.k_neighbors)
                .map_err(|e| format!("Matching failed: {:?}", e))?
        } else {
            let use_mahalanobis = args.matching_method == "mahalanobis";
            engine
                .run_matching(args.k_neighbors, use_mahalanobis)
                .map_err(|e| format!("Matching failed: {:?}", e))?
        };
        if let Some(path) = args.output_json {
            let json = serde_json::to_string(&weights)?;
            std::fs::write(path, json)?;
        } else {
            println!("Matching completed. Generated {} weights.", weights.len());
            println!(
                "First 10 weights: {:?}",
                weights.iter().take(10).collect::<Vec<_>>()
            );
        }
    } else {
        return Err(format!("Unknown analysis type: {}", args.analysis_type).into());
    }
    Ok(())
}

fn run_report(args: ReportArgs) -> Result<(), Box<dyn Error>> {
    let df = LazyCsvReader::new(&args.data)
        .with_has_header(true)
        .finish()?
        .collect()?;
    let predictors: Vec<&str> = args.predictors.iter().map(AsRef::as_ref).collect();
    let categorical_predictors: Vec<&str> = args
        .categorical
        .as_ref()
        .map(|v| v.iter().map(AsRef::as_ref).collect())
        .unwrap_or_else(Vec::new);
    let results = OaxacaBuilder::new(df, &args.outcome, &args.group, &args.reference)
        .predictors(&predictors)
        .categorical_predictors(&categorical_predictors)
        .run()?;
    let summary = results.get_summary_table();
    let detailed = results.get_detailed_table();
    let overall_data = summary
        .into_iter()
        .map(|(name, values)| OverallRow {
            name: name.clone(),
            coefficient: *values.estimate(),
            std_error: *values.std_err(),
            t_stat: *values.t_stat(),
            p_value: *values.p_value(),
        })
        .collect();
    let detailed_data = detailed
        .into_iter()
        .map(|(variable, explained, unexplained)| DetailedRow {
            variable,
            explained,
            unexplained,
        })
        .collect();
    let report_data = ReportData {
        total_gap: format!("{:.4}", results.total_gap()),
        explained_gap: format!("{:.4}", results.explained().estimate()),
        unexplained_gap: format!("{:.4}", results.unexplained().estimate()),
        overall: overall_data,
        detailed: detailed_data,
    };
    let html = report_data.render()?;
    std::fs::write(&args.output, html)?;
    println!("Report successfully generated at: {}", args.output.display());
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Commands::Run(args)) => run_analysis(args),
        Some(Commands::Report(args)) => run_report(args),
        None => run_analysis(cli.run_args),
    };
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        let mut cmd = Cli::command();
        let _ = cmd.print_help();
        std::process::exit(1);
    }
}
