import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
train_data = pd.read_csv('./drugLibTrain_raw.tsv', sep='\t')
test_data = pd.read_csv('./drugLibTest_raw.tsv', sep='\t')

# Inspect the first few rows to understand the structure
print(train_data.head())
print(test_data.head())

# Check for missing values
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# Filter out rows where important columns like 'urlDrugName', 'benefitsReview', 'rating', etc. are missing
train_data_cleaned = train_data.dropna(subset=['urlDrugName', 'benefitsReview', 'rating'])
test_data_cleaned = test_data.dropna(subset=['urlDrugName', 'benefitsReview', 'rating'])

# Optionally filter by conditions or specific ratings (e.g., effectiveness, side effects)
train_data_cleaned = train_data_cleaned[train_data_cleaned['sideEffects'].notnull()]
test_data_cleaned = test_data_cleaned[test_data_cleaned['sideEffects'].notnull()]

# Check for duplicates
print(f"Duplicate rows in train data: {train_data_cleaned.duplicated().sum()}")
print(f"Duplicate rows in test data: {test_data_cleaned.duplicated().sum()}")

# Drop duplicates if any
train_data_cleaned = train_data_cleaned.drop_duplicates()
test_data_cleaned = test_data_cleaned.drop_duplicates()

# Standardize data types (e.g., convert 'rating' to numeric)
train_data_cleaned['rating'] = pd.to_numeric(train_data_cleaned['rating'], errors='coerce')
test_data_cleaned['rating'] = pd.to_numeric(test_data_cleaned['rating'], errors='coerce')

# Check for inconsistent values in the 'sideEffectsReview' column
print(train_data_cleaned['sideEffectsReview'].unique())
print(test_data_cleaned['sideEffectsReview'].unique())

# Standardize text values (e.g., convert to lowercase)
train_data_cleaned['sideEffectsReview'] = train_data_cleaned['sideEffectsReview'].str.lower()
test_data_cleaned['sideEffectsReview'] = test_data_cleaned['sideEffectsReview'].str.lower()

# Remove extreme outliers in 'rating' (e.g., ratings outside 1-5 range)
train_data_cleaned = train_data_cleaned[(train_data_cleaned['rating'] >= 1) & (train_data_cleaned['rating'] <= 5)]
test_data_cleaned = test_data_cleaned[(test_data_cleaned['rating'] >= 1) & (test_data_cleaned['rating'] <= 5)]

# Text Preprocessing function for reviews
def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    else:
        return ""  # Return an empty string if the value is not a string (e.g., NaN or float)

# Apply text preprocessing to the 'benefitsReview' and 'commentsReview' columns
train_data_cleaned['benefitsReview'] = train_data_cleaned['benefitsReview'].apply(preprocess_text)
test_data_cleaned['benefitsReview'] = test_data_cleaned['benefitsReview'].apply(preprocess_text)
train_data_cleaned['commentsReview'] = train_data_cleaned['commentsReview'].apply(preprocess_text)
test_data_cleaned['commentsReview'] = test_data_cleaned['commentsReview'].apply(preprocess_text)

# Save the cleaned datasets (optional)
train_data_cleaned.to_csv('./final_cleaned_train_data.csv', index=False)
test_data_cleaned.to_csv('./final_cleaned_test_data.csv', index=False)

# Display cleaned data
print(train_data_cleaned.head())
print(test_data_cleaned.head())
