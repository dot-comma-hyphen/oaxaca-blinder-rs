use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_mean_decomposition() {
    let mut cmd = Command::cargo_bin("oaxaca-cli").unwrap();
    cmd.arg("--data")
        .arg("tests/data/wage.csv")
        .arg("--outcome")
        .arg("wage")
        .arg("--group")
        .arg("gender")
        .arg("--reference")
        .arg("F")
        .arg("--predictors")
        .arg("education");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Oaxaca-Blinder Decomposition Results"))
        .stdout(predicate::str::contains("Two-Fold Decomposition"))
        .stdout(predicate::str::contains("Detailed Decomposition (Explained)"))
        .stdout(predicate::str::contains("Detailed Decomposition (Unexplained)"));
}

#[test]
fn test_mean_decomposition_with_categorical() {
    let mut cmd = Command::cargo_bin("oaxaca-cli").unwrap();
    cmd.arg("--data")
        .arg("tests/data/wage.csv")
        .arg("--outcome")
        .arg("wage")
        .arg("--group")
        .arg("gender")
        .arg("--reference")
        .arg("F")
        .arg("--predictors")
        .arg("education")
        .arg("--categorical")
        .arg("gender");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Oaxaca-Blinder Decomposition Results"));
}

#[test]
fn test_quantile_decomposition() {
    let mut cmd = Command::cargo_bin("oaxaca-cli").unwrap();
    cmd.arg("--data")
        .arg("tests/data/wage.csv")
        .arg("--outcome")
        .arg("wage")
        .arg("--group")
        .arg("gender")
        .arg("--reference")
        .arg("F")
        .arg("--predictors")
        .arg("education")
        .arg("--analysis-type")
        .arg("quantile");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Machado-Mata Quantile Decomposition Results"));
}

#[test]
fn test_invalid_argument() {
    let mut cmd = Command::cargo_bin("oaxaca-cli").unwrap();
    cmd.arg("--data")
        .arg("tests/data/non_existent_file.csv")
        .arg("--outcome")
        .arg("wage")
        .arg("--group")
        .arg("gender")
        .arg("--reference")
        .arg("F")
        .arg("--predictors")
        .arg("education");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}
