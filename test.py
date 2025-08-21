import pandas as pd
import numpy as np

data = {
    "Order ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    
    "Customer Name": ["Alice", "Bob", "Charlie", "David", "Eve"] * 4,
    
    # Added some nulls and inconsistent values
    "Category": ["Furniture", "Office Supplies", "Technology", np.nan, "Furniture",
                 "Technology", "Office Supplies", "Furniture", "Technology", "Furniture",
                 "Office Supplies", "Technology", "Furniture", np.nan, "Furniture",
                 "Office Supplies", "Technology", "Furniture", "Office Supplies", "Technology"],
    
    # Add big numbers for scaling + some None for missing
    "Amount": [200, 450, None, 300, 500,
               700, 200, 450, None, 300,
               1200, 450, 200, 500, None,
               350, 450, 200, 600, 100000],   # <-- outlier
    
    # Include negative values + missing + large positive outlier
    "Profit": [20, None, 50, -10, 80,
               100, None, 40, 60, -20,
               200, None, 30, 50, 70,
               None, 40, 10, -5, 1000],       # <-- outlier
    
    # Add nulls + variety in ranges
    "Quantity": [1, 2, None, 1, 3,
                 5, 2, None, 4, 1,
                 6, 3, 1, 2, None,
                 2, 2, 1, 5, 50],             # <-- outlier
    
    "Order Date": pd.date_range(start="2021-01-01", periods=20, freq="D"),
}

df = pd.DataFrame(data)

# ✅ Add duplicates (copy rows 2 and 5)
df = pd.concat([df, df.iloc[[1, 4]]], ignore_index=True)


# Save as CSV
df.to_csv("messy_dataset.csv", index=False)

print("✅ Messy dataset created and saved as 'messy_dataset.csv'")
print(df.head(10))
