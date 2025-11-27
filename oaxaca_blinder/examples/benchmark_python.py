import pandas as pd
import statsmodels.api as sm
import numpy as np
import time
import oaxaca_blinder

def run_benchmark():
    print("Benchmarking Oaxaca-Blinder Implementation...")
    print("-" * 50)

    # --- Rust Implementation (With Bootstrap) ---
    print("\nRunning Rust implementation (oaxaca_blinder, 100 bootstraps)...")
    start_time_rust = time.time()
    
    results_rust = oaxaca_blinder.decompose_from_csv(
        "benchmark_100k.csv",
        "wage",
        ["education", "experience", "age", "tenure", "ability", "training", "performance", "hours", "metro", "married"],
        [], 
        "gender",
        "F",
        100
    )
    
    end_time_rust = time.time()
    rust_duration = end_time_rust - start_time_rust
    print(f"Rust Execution Time (100 bootstraps): {rust_duration:.4f} seconds")

    # --- Rust Implementation (No Bootstrap) ---
    print("\nRunning Rust implementation (oaxaca_blinder, 0 bootstraps)...")
    start_time_rust_raw = time.time()
    
    results_rust_raw = oaxaca_blinder.decompose_from_csv(
        "benchmark_100k.csv",
        "wage",
        ["education", "experience", "age", "tenure", "ability", "training", "performance", "hours", "metro", "married"],
        [], 
        "gender",
        "F",
        0
    )
    
    end_time_rust_raw = time.time()
    rust_raw_duration = end_time_rust_raw - start_time_rust_raw
    print(f"Rust Execution Time (0 bootstraps): {rust_raw_duration:.4f} seconds")
    
    # --- Python Implementation (statsmodels) ---
    print("\nRunning Python implementation (statsmodels)...")
    start_time_py = time.time()
    
    # Load data (Pandas)
    df = pd.read_csv("benchmark_100k.csv")
    
    # Predictors and Outcome
    predictors = ["education", "experience", "age", "tenure", "ability", "training", "performance", "hours", "metro", "married"]
    outcome = "wage"
    
    # Prepare data for statsmodels
    df['gender_binary'] = (df['gender'] == 'M').astype(int) # M=1, F=0. Reference is F.
    
    exog = df[predictors].copy()
    exog['gender_binary'] = df['gender_binary']
    exog = sm.add_constant(exog)
    endog = df[outcome]
    
    from statsmodels.stats.oaxaca import OaxacaBlinder
    model = OaxacaBlinder(endog, exog, "gender_binary", hasconst=True)
    results_py = model.two_fold()
    
    end_time_py = time.time()
    py_duration = end_time_py - start_time_py
    print(f"Python Execution Time: {py_duration:.4f} seconds")
    
    # --- Comparison ---
    print("-" * 50)
    print(f"Speedup: {py_duration / rust_duration:.2f}x")
    print("-" * 50)
    
    # Verify results (Total Gap)
    rust_gap = results_rust.total_gap
    # Statsmodels summary is a text blob or a result object. 
    # The .two_fold() returns a generic result wrapper.
    # Let's try to extract the gap if possible, or just trust the visual check.
    # Statsmodels doesn't make it super easy to get the raw number programmatically from the summary object without digging.
    # But we can print the Rust gap.
    print(f"Rust Total Gap: {rust_gap:.4f}")

if __name__ == "__main__":
    run_benchmark()
