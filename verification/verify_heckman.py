import pandas as pd
import numpy as np
from oaxaca_blinder import OaxacaBlinder

def create_selection_data():
    np.random.seed(42)
    n = 2000
    
    # Predictors
    education = np.random.normal(12, 2, n)
    experience = np.random.normal(10, 5, n)
    # Selection instrument (e.g., family size, not in wage eq)
    family_size = np.random.poisson(2, n)
    
    # Selection process (Probit latent)
    # Group A (e.g., Men) more likely to work
    z_a = -1.0 + 0.1*education + 0.05*experience - 0.2*family_size + np.random.normal(0, 1, n)
    participation_a = (z_a > 0).astype(int)
    
    # Group B (e.g., Women) less likely
    z_b = -0.5 + 0.1*education + 0.05*experience - 0.5*family_size + np.random.normal(0, 1, n)
    participation_b = (z_b > 0).astype(int)
    
    # Wage process (observed only if participation=1)
    wage = 5.0 + 0.5*education + 0.2*experience + np.random.normal(0, 2, n)
    # Add gap
    wage[n//2:] = wage[n//2:] - 2.0 # Artificial gap for Group B
    
    group = ['GroupA'] * (n//2) + ['GroupB'] * (n - n//2)
    participation = np.concatenate([participation_a[:n//2], participation_b[n//2:]])
    
    df = pd.DataFrame({
        'wage': wage,
        'group': group,
        'education': education,
        'experience': experience,
        'family_size': family_size,
        'employed': participation
    })
    
    # Set wage to NaN if not employed
    df.loc[df['employed'] == 0, 'wage'] = np.nan
    
    return df

def test_heckman():
    print("Testing Heckman Decomposition...")
    df = create_selection_data()
    
    model = OaxacaBlinder(
        df,
        outcome='wage',
        group='group',
        reference_group='GroupA',
        predictors=['education', 'experience'],
        selection_outcome='employed',
        selection_predictors=['education', 'experience', 'family_size']
    )
    
    results = model.fit()
    print("Heckman model fitted successfully.")
    
    # Check detailed selection results
    # Assuming PyOaxacaResults -> PyTwoFoldResults -> detailed_selection
    det_sel = results.two_fold.detailed_selection
    print(f"Detailed Selection Components: {len(det_sel)}")
    for comp in det_sel:
        print(f"  {comp.name}: {comp.estimate:.4f}")
        
    assert len(det_sel) > 0, "Expected detailed selection components"


if __name__ == "__main__":
    test_heckman()
