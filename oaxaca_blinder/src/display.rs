#[cfg(feature = "display")]
use comfy_table::{Cell, Table};

use crate::types::OaxacaResults;

#[cfg(feature = "display")]
impl OaxacaResults {
    /// Prints a formatted summary of the decomposition results to the console.
    pub fn summary(&self) {
        println!("Oaxaca-Blinder Decomposition Results");
        println!("========================================");
        println!("Group A (Advantaged): {} observations", self.n_a);
        println!("Group B (Reference):  {} observations", self.n_b);
        println!("Total Gap: {:.4}", self.total_gap);
        println!();

        let mut two_fold_table = Table::new();
        two_fold_table.set_header(vec![
            "Component",
            "Estimate",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.aggregate() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            two_fold_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("Two-Fold Decomposition");
        println!("{}", two_fold_table);

        let mut explained_table = Table::new();
        explained_table.set_header(vec![
            "Variable",
            "Contribution",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.detailed_explained() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            explained_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("\nDetailed Decomposition (Explained)");
        println!("{}", explained_table);

        let mut unexplained_table = Table::new();
        unexplained_table.set_header(vec![
            "Variable",
            "Contribution",
            "Std. Err.",
            "p-value",
            "95% CI",
        ]);
        for component in self.two_fold.detailed_unexplained() {
            let ci = format!("[{:.3}, {:.3}]", component.ci_lower(), component.ci_upper());
            unexplained_table.add_row(vec![
                Cell::new(component.name()),
                Cell::new(format!("{:.4}", component.estimate())),
                Cell::new(format!("{:.4}", component.std_err())),
                Cell::new(format!("{:.4}", component.p_value())),
                Cell::new(ci),
            ]);
        }
        println!("\nDetailed Decomposition (Unexplained)");
        println!("{}", unexplained_table);
    }
}

impl OaxacaResults {
    /// Exports the results to a LaTeX table fragment.
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();
        latex.push_str("\\begin{table}[ht]\n");
        latex.push_str("\\centering\n");
        latex.push_str("\\begin{tabular}{lcccc}\n");
        latex.push_str("\\hline\n");
        latex.push_str("Component & Estimate & Std. Err. & p-value & 95\\% CI \\\\\n");
        latex.push_str("\\hline\n");
        latex.push_str("\\multicolumn{5}{l}{\\textit{Two-Fold Decomposition}} \\\\\n");

        for component in self.two_fold.aggregate() {
            latex.push_str(&format!(
                "{} & {:.4} & {:.4} & {:.4} & [{:.3}, {:.3}] \\\\\n",
                component.name(),
                component.estimate(),
                component.std_err(),
                component.p_value(),
                component.ci_lower(),
                component.ci_upper()
            ));
        }
        latex.push_str("\\hline\n");
        latex.push_str("\\end{tabular}\n");
        latex.push_str("\\caption{Oaxaca-Blinder Decomposition Results}\n");
        latex.push_str("\\label{tab:oaxaca_results}\n");
        latex.push_str("\\end{table}\n");
        latex

    }

    /// Exports the results to a Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("### Oaxaca-Blinder Decomposition Results\n\n");
        md.push_str("| Component | Estimate | Std. Err. | p-value | 95% CI |\n");
        md.push_str("|---|---|---|---|---|\n");

        for component in self.two_fold.aggregate() {
            md.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | [{:.3}, {:.3}] |\n",
                component.name(),
                component.estimate(),
                component.std_err(),
                component.p_value(),
                component.ci_lower(),
                component.ci_upper()
            ));
        }
        md
    }

    /// Exports the results to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}