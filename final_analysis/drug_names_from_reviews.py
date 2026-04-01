# Import necessary libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the drug review dataset (replace with the actual dataset ID for drug reviews)
drug_reviews = fetch_ucirepo(id=461)  # Example ID for drug reviews dataset

# Extract unique drug names from the 'urlDrugName' column
drug_names = drug_reviews.data.features['urlDrugName'].unique()

# Print all unique drug names
print(drug_names)

# Optionally, save the drug names to a CSV file
pd.Series(drug_names).to_csv('drug_names.csv', index=False, header=True)
