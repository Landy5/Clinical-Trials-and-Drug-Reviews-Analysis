import requests
import time


# Function to download CSV data for a single drug
def download_clinical_trials_csv(drug_name):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "format": "csv",  # CSV format
        "query.term": drug_name
    }

    # Make the API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        # Save the CSV data to a file named after the drug
        filename = f"{drug_name.lower()}.csv"
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Data for {drug_name} saved to {filename}")
    else:
        print(f"Failed to fetch data for {drug_name}: {response.status_code}")


# Function to loop through the list of drugs and download data
def process_drugs(drugs):
    for drug in drugs:
        print(f"Pulling data for {drug}...")
        download_clinical_trials_csv(drug)
        # To avoid overwhelming the API, introduce a delay between requests
        time.sleep(2)


# List of drugs to pull data for
drugs = ["Treprostinil", "Estradiol", "Ibuprofen", "Metformin", "Atorvastatin"]

# Run the data processing for all drugs
process_drugs(drugs)
