import requests
from lxml import html
import pandas as pd
import re

def scrape_reviews(url, pages=1):
    reviews_list = []
    
    # Loop over the pages
    for page in range(1, pages + 1):
        page_url = f"{url}?page={page}"
        response = requests.get(page_url)
        tree = html.fromstring(response.content)

        # Use the correct XPath based on your inspection
        review_blocks = tree.xpath('//div[contains(@class, "review-details")]')

        # Iterate over the reviews
        for block in review_blocks:
            review_text = block.text_content().strip()

            # Extract data with regex
            try:
                date = re.search(r'\d{1,2}/\d{1,2}/\d{4}', review_text).group()
            except AttributeError:
                date = None

            try:
                overall_rating = re.search(r'Overall rating\s+(\d+.\d+)', review_text).group(1)
            except AttributeError:
                overall_rating = None

            try:
                effectiveness = re.search(r'Effectiveness\s+(\d+.\d+)', review_text).group(1)
            except AttributeError:
                effectiveness = None

            try:
                ease_of_use = re.search(r'Ease of Use\s+(\d+.\d+)', review_text).group(1)
            except AttributeError:
                ease_of_use = None

            try:
                satisfaction = re.search(r'Satisfaction\s+(\d+.\d+)', review_text).group(1)
            except AttributeError:
                satisfaction = None

            try:
                textual_review = review_text.split('Satisfaction')[-1].strip()
            except IndexError:
                textual_review = None

            review_data = {
                'Date': date,
                'Overall Rating': overall_rating,
                'Effectiveness': effectiveness,
                'Ease of Use': ease_of_use,
                'Satisfaction': satisfaction,
                'Textual Review': textual_review
            }
            reviews_list.append(review_data)

    return reviews_list

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# URL of the drug reviews
url = 'https://reviews.webmd.com/drugs/drugreview-64439-abilify-oral'
pages = 20 # Number of pages to scrape

# Scrape the reviews
reviews = scrape_reviews(url, pages)

# Save to CSV
csv_file_name = 'abilify_oral_data.csv'  # Use a valid local path if needed
save_to_csv(reviews, csv_file_name)

