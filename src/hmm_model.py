import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

def fit_hmm_model(df):
    data = df.copy()
    features = data[['SPY_RETURN', 'VOLATILITY']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=0)
    
    try:
        model.fit(features_scaled)
        probs = model.predict_proba(features_scaled)
        preds = model.predict(features_scaled)
        
        # Label switching logic
        vol_mean_state_0 = np.mean(features[preds == 0, 1])
        vol_mean_state_1 = np.mean(features[preds == 1, 1])
        
        if vol_mean_state_1 > vol_mean_state_0:
            risky_idx, safe_idx = 1, 0
        else:
            risky_idx, safe_idx = 0, 1
            
        data['Prob_Safe'] = probs[:, safe_idx]
        data['Prob_Risky'] = probs[:, risky_idx]
        
    except ValueError:
        # If the model fails to converge, assign neutral probabilities
        data['Prob_Safe'] = 0.5
        data['Prob_Risky'] = 0.5
        
    return data

if __name__ == "__main__":
    try:
        from .data_loader import download_and_process_data
    except ImportError:
        from data_loader import download_and_process_data
    
    df_market = download_and_process_data()
    df_results = fit_hmm_model(df_market)

    print(df_results[['SPY_RETURN', 'VOLATILITY', 'Prob_Safe', 'Prob_Risky']].sample(10))

    
