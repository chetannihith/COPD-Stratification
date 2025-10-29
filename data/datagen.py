import numpy as np
import pandas as pd
import os

# Set parameters
N = 2000
total_time = 6  # Total time for spirometry test (seconds)
num_points = 51
dt = total_time / (num_points - 1)  # Time interval between points

# Generate demographic data
age = np.random.uniform(40, 70, N)
sex = np.random.binomial(1, 0.5, N)  # 0: female, 1: male
smoking = np.random.binomial(1, 0.2, N)  # Smoking status

# Compute COPD probability and labels
P_COPD = np.where(smoking == 1, 0.2, 0.02)
COPD = np.random.binomial(1, P_COPD)

# Compute predicted FVC based on age and sex
FVC_pred = np.where(
    sex == 1,
    5.0 - 0.02 * (age - 40),  # Male
    4.0 - 0.02 * (age - 40)   # Female
)

# Compute actual FVC with variation
FVC = np.where(
    COPD == 0,
    FVC_pred * (0.9 + 0.2 * np.random.rand(N)),  # Healthy
    FVC_pred * (0.6 + 0.2 * np.random.rand(N))   # COPD
)
FVC_mL = FVC * 1000

# Set k based on COPD status
k = np.where(COPD == 0, 12, 5.7)

# Generate flow sequences
t = np.arange(num_points) / (num_points - 1) * total_time
V = FVC_mL[:, np.newaxis] * (1 - np.exp(-k[:, np.newaxis] * t)) / (1 - np.exp(-k[:, np.newaxis]))
V_int = V.round().astype(int)

# Compute FEV1
FEV1_idx = int(1 / dt) if dt > 0 else 0
FEV1 = V[:, FEV1_idx] / 1000

# Compute PEF
dV = np.diff(V_int, axis=1) / dt
flow_mL_s = dV
PEF_mL_s = np.max(flow_mL_s, axis=1)
PEF_L_min = PEF_mL_s * (60 / 1000)  # Convert mL/s to L/min

# Convert flow to comma-separated strings
flow_str = [','.join(map(str, row)) for row in V_int]

# Create DataFrame
data = {
    'age': age,
    'sex': sex,
    'smoking': smoking,
    'COPD': COPD,
    'FVC': FVC,
    'FEV1': FEV1,
    'PEF': PEF_L_min,
    'flow': flow_str
}
df = pd.DataFrame(data)

os.makedirs('./data/train', exist_ok=True)
df.to_excel('./data/train/synthetic_data.xlsx', index=False)