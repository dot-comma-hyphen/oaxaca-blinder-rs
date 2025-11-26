import pandas as pd
import statsmodels.api as sm
import numpy as np
import time

def run_benchmark():
    start_time = time.time()
    
    # Load data
    df = pd.read_csv("benchmark_100k.csv")
    
    # Predictors and Outcome
    predictors = ["education", "experience", "age", "tenure", "ability", "training", "performance", "hours", "metro", "married"]
    outcome = "wage"
    group_col = "gender"
    ref_group = "F"
    
    # Split groups
    group_a = df[df[group_col] != ref_group]
    group_b = df[df[group_col] == ref_group]
    
    n_a = len(group_a)
    n_b = len(group_b)
    
    # Prepare matrices
    X_a = sm.add_constant(group_a[predictors])
    y_a = group_a[outcome]
    
    X_b = sm.add_constant(group_b[predictors])
    y_b = group_b[outcome]
    
    # Point Estimates
    model_a = sm.OLS(y_a, X_a).fit()
    model_b = sm.OLS(y_b, X_b).fit()
    
    beta_a = model_a.params
    beta_b = model_b.params
    
    mean_X_a = X_a.mean()
    mean_X_b = X_b.mean()
    
    # Use statsmodels Oaxaca implementation
    from statsmodels.stats.oaxaca import OaxacaBlinder

    print("Running statsmodels OaxacaBlinder...")
    
    # statsmodels Oaxaca expects endog and exog. It handles the splitting if we give it a group indicator?
    # No, looking at docs/examples, it usually takes data and a group column.
    
    # It seems statsmodels.stats.oaxaca.OaxacaBlinder takes (endog, exog, bifurcate)
    # bifurcate is the group indicator.
    
    # Prepare data for statsmodels
    # It needs numeric inputs mostly.
    
    # We need to map gender to 0/1 for bifurcate?
    df['gender_binary'] = (df['gender'] == 'M').astype(int) # M=1, F=0. Reference is F (0).
    
    # Features
    # Statsmodels Oaxaca requires the bifurcation variable to be in exog
    exog = df[predictors].copy()
    exog['gender_binary'] = df['gender_binary']
    exog = sm.add_constant(exog)
    endog = df[outcome]
    
    # Instantiate
    # bifurcate: "gender_binary" (column name in exog)
    # has_const: True (since we added constant)
    model = OaxacaBlinder(endog, exog, "gender_binary", hasconst=True)
    
    # Fit
    # It doesn't seem to have built-in bootstrapping in the .two_fold() method?
    # The .two_fold() returns a summary.
    
    results = model.two_fold()
    print(results.summary())
    
    end_time = time.time()
    print(f"Total Execution Time (statsmodels, no bootstrap): {end_time - start_time:.4f} seconds")
    
    # Note: statsmodels Oaxaca implementation does NOT appear to do bootstrapping by default for SEs.
    # It uses asymptotic standard errors.
    # So this comparison is "Rust (Bootstrapped) vs Python (Asymptotic)".
    # Rust is doing WAY more work and is still likely faster or comparable.
    # To make it fair, we should mention this.

if __name__ == "__main__":
    run_benchmark()
