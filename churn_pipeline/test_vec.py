import numpy as np

# Mock SHAP values for 3 records, 2 features
sv = np.array([
    [0.1, -0.05],   # risk_sum=0.1, safe_sum=0.05
    [-0.2, -0.1],   # risk_sum=0.0, safe_sum=0.3
    [0.0, 0.0]      # risk_sum=0.0, safe_sum=0.0
])

norm_scores = np.array([0.8, 0.6, 0.5])

risk_sums = np.sum(np.where(sv > 0, sv, 0), axis=1)
safe_sums = np.sum(np.abs(np.where(sv < 0, sv, 0)), axis=1)

total_abs_sums = risk_sums + safe_sums

with np.errstate(divide='ignore', invalid='ignore'):
    risk_ratios = np.where(total_abs_sums > 0, risk_sums / total_abs_sums, 0.5)

factors = np.ones_like(risk_ratios)
low_mask = risk_ratios < 0.5
factors[low_mask] = 0.20 + 1.20 * risk_ratios[low_mask]
high_mask = ~low_mask
factors[high_mask] = 1.0 + 0.30 * (risk_ratios[high_mask] - 0.5)

factors[total_abs_sums == 0] = 1.0

adj_scores = np.clip(norm_scores * factors, 0, 1)

print("risk_ratios:", risk_ratios)
print("factors:", factors)
print("adj_scores:", adj_scores)
