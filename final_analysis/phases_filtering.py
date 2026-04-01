import pandas as pd

# Load the data from the CSV file
file_path = './clinical_trials_for_drugs.csv'  # Correct file path from the upload
clinical_trials_df = pd.read_csv(file_path)

# Inspect the columns and check the first few rows to understand the structure
print(clinical_trials_df.columns)
print(clinical_trials_df.head())

# Normalize the phases column (assuming variations in case like 'PHASE3', 'Phase3', etc.)
clinical_trials_df['phases'] = clinical_trials_df['phases'].str.upper()

# Filter the data for Phase 3 and Phase 4 trials
clinical_trials_filtered = clinical_trials_df[clinical_trials_df['phases'].isin(['PHASE3', 'PHASE4'])]

# Check for missing values and handle them (if any)
print(clinical_trials_filtered.isnull().sum())

# Save the cleaned and filtered data (optional)
cleaned_file_path = './cleaned_clinical_trials.csv'
clinical_trials_filtered.to_csv(cleaned_file_path, index=False)

# Output the filtered data
print(clinical_trials_filtered.head())
