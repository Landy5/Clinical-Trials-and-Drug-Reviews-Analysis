import requests
import csv
import time

# Function to make API requests for a single drug's clinical trials
def get_clinical_trials(drug_name):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "format": "json",
        "query.term": drug_name,
        "filter.overallStatus": "COMPLETED",  # Modify this based on your need
        "pageSize": 100  # Number of results to fetch per request
    }
    
    headers = {
        "accept": "application/json"
    }

    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {drug_name}: {response.status_code}")
        return None

# Function to save data to a CSV file
def save_to_csv(data, filename):
    if not data:
        print("No data to save.")
        return
    
    # Get all unique fieldnames from the data to ensure the CSV has all columns
    fieldnames = set()
    for entry in data:
        fieldnames.update(entry.keys())

    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# Main function to loop through drugs and get their clinical trial data
def pull_data_for_drugs(drugs):
    all_data = []
    for drug in drugs:
        print(f"Pulling data for {drug}...")
        drug_data = get_clinical_trials(drug)
        if drug_data:
            # Assume the clinical trial results are under the 'studies' key
            studies = drug_data.get("studies", [])
            all_data.extend(studies)
        
        # To avoid overwhelming the API, introduce a delay between requests
        time.sleep(2)

    # Save all data to CSV
    save_to_csv(all_data, "clinical_trials_data.csv")
    print("Data saved to clinical_trials_data.csv")

# List of drugs to pull data for
drugs = ["Treprostinil", "Estradiol", "Ibuprofen", "Metformin", "Atorvastatin"]

# Run the data pull and save to CSV
pull_data_for_drugs(drugs)
