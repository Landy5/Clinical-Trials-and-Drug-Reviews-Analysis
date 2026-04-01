import pandas as pd

# Generate the URL for each drug based on its name
def generate_drug_url(drug_name):
    base_url = "https://reviews.webmd.com/drugs/drugreview-"
    drug_name = drug_name.lower().replace(" ", "-")
    drug_id = "64439"  # Adjust this if required for each drug
    return f"{base_url}{drug_id}-{drug_name}"

# Save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
