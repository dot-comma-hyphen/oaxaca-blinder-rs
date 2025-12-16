import pandas as pd
import numpy as np
from oaxaca_blinder import OaxacaBlinder

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

def test_interpretation():
    print("Testing Interpretation...")
    df = create_dummy_data()
    model = OaxacaBlinder(
        df,
        outcome='wage',
        group='gender',
        reference_group='Male',
        predictors=['education', 'experience']
    )
    results = model.fit()
    
    interpretation = results.interpret()
    print("\nInterpretation Output:")
    print("-" * 20)
    print(interpretation)
    print("-" * 20)
    
    assert "The total gap is" in interpretation
    assert "% of this gap is explained" in interpretation

if __name__ == "__main__":
    test_interpretation()
