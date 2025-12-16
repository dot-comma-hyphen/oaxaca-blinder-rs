import pandas as pd
import numpy as np
from oaxaca_blinder import OaxacaBlinder, run_dfl_from_csv
import matplotlib.pyplot as plt
import polars as pl

def create_dummy_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'wage': np.random.normal(20, 5, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'education': np.random.normal(12, 2, n),
        'experience': np.random.normal(10, 5, n)
    })
    # Add some structure
    df.loc[df['gender']=='Male', 'wage'] += 5
    return df

def test_oaxaca_plot():
    print("Testing Interpretation...")
    df = create_dummy_data()
    df_pl = pl.from_pandas(df)
    model = OaxacaBlinder(
        df_pl,
        outcome='wage',
        group='gender',
        reference_group='Male',
        predictors=['education', 'experience']
    )
    results = model.fit()
    
    # Test Plot
    try:
        fig = results.plot(kind='bar')
        print("Plot generated successfully.")
        # plt.show() # Uncomment to see plot
    except Exception as e:
        print(f"Plotting failed: {e}")

def test_dfl_plot():
    print("\nTesting DFL Plot...")
    # Create a dummy CSV for DFL
    df = create_dummy_data()
    df.to_csv("dummy_dfl.csv", index=False)
    
    try:
        results = run_dfl_from_csv(
            "dummy_dfl.csv",
            outcome='wage',
            group='gender',
            reference_group='Male',
            predictors=['education', 'experience']
        )
        fig = results.plot()
        print("DFL Plot generated successfully.")
        # plt.show()
    except Exception as e:
        print(f"DFL Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_oaxaca_plot()
    test_dfl_plot()
