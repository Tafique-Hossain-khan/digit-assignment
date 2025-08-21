import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 400

# Features
age = np.random.randint(18, 65, n)  # age between 18 and 65
sex = np.random.choice(["male", "female"], n)
bmi = np.round(np.random.normal(30, 6, n), 1)  # mean BMI ~30
children = np.random.randint(0, 5, n)  # 0 to 4 children
smoker = np.random.choice(["yes", "no"], n, p=[0.2, 0.8])  # ~20% smokers
region = np.random.choice(["northeast", "northwest", "southeast", "southwest"], n)

# Target (charges) - synthetic formula with noise
charges = (
    250 * age +
    300 * children +
    500 * (bmi - 25) +
    np.where(sex == "male", 200, -200) +
    np.where(smoker == "yes", 12000, 0) +
    np.random.normal(0, 2000, n)  # noise
)

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": np.round(charges, 2)
})

# Save to CSV
df.to_csv("insurance_data.csv", index=False)

print("✅ insurance_data.csv generated with shape:", df.shape)

