import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ['SPY_RETURN', 'VOLATILITY']
STATE_COUNT = 3
NEUTRAL_PROBABILITY = 1 / STATE_COUNT


def _state_order_by_volatility(features, predicted_states):
    """Return state indices ordered from lowest to highest realized volatility."""
    volatility_idx = 1
    vol_means = []

    for state in range(STATE_COUNT):
        state_mask = predicted_states == state
        if state_mask.any():
            vol_means.append(features[state_mask, volatility_idx].mean())
        else:
            vol_means.append(np.inf)

    return np.argsort(vol_means)


def fit_hmm_3states(df):
    """Fit a 3-state Gaussian HMM and label states by relative volatility.

    The three states are interpreted as market regimes:
        - LOW  volatility  -> calm "bull" regime
        - MID  volatility  -> "sideways" regime
        - HIGH volatility  -> turbulent "bear" regime

    We use "full" covariance, the same setting as the 2-state model. The only structural
    difference between them is the number of states, which makes the
    comparison cleaner. 
    """
    data = df.copy()
    features = data[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = hmm.GaussianHMM(
        n_components=STATE_COUNT,
        covariance_type="full",
        n_iter=100,
        random_state=0,
    )

    try:
        model.fit(features_scaled)
        probs = model.predict_proba(features_scaled)
        preds = model.predict(features_scaled)

        # Label switching logic for 3 states:
        # sort regimes from the calmest to the most volatile.
        order = _state_order_by_volatility(features, preds)
        low_idx, mid_idx, high_idx = order[0], order[1], order[2]

        data['Prob_Low'] = probs[:, low_idx]
        data['Prob_Mid'] = probs[:, mid_idx]
        data['Prob_High'] = probs[:, high_idx]

    except ValueError:
        # If the model fails to converge, assign neutral probabilities.
        data['Prob_Low'] = NEUTRAL_PROBABILITY
        data['Prob_Mid'] = NEUTRAL_PROBABILITY
        data['Prob_High'] = NEUTRAL_PROBABILITY

    return data
