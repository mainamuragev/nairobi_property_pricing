# Nairobi Property Pricing Project
# Nairobi Property Pricing Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete data pipeline for scraping, cleaning, enriching, and analyzing property listings in Nairobi. This project transforms raw web data into actionable insights, including price normalization, bedroom extraction, and the creation of an interactive affordability map.

##  Overview

This project automates the collection and processing of Nairobi real estate data. It is designed to answer key questions like: "What is the average rent per bedroom in different areas?" and "Where are the most affordable neighborhoods based on my budget?".

The pipeline consists of several stages:
1.  **Scraping**: Collecting raw listing data from property websites.
2.  **Cleaning & Parsing**: Normalizing prices and intelligently extracting the number of bedrooms from messy titles and URLs.
3.  **Enrichment**: Calculating affordability metrics like price per bedroom.
4.  **Analysis & Visualization**: Generating location-based summaries and an interactive map.

##  Key Features

*   **Automated Data Pipeline**: A series of Python scripts that handle the entire data workflow from `scrape` to `analyze`.
*   **Intelligent Bedroom Parsing**: Extracts bedroom counts from unstructured text (e.g., "2br", "2 bedroom", "2 bed") using the `parser.py` module.
*   **Price Normalization**: Cleans and standardizes price formats (e.g., "Ksh 45,000", "45k") into numerical values for analysis.
*   **Affordability Calculation**: Computes the **price per bedroom**, a key metric for comparing value across different property sizes.
*   **Location Summaries**: Aggregates data by area to generate summary statistics like average rent and bedroom counts (`location_summary_clean.csv`).
*   **Interactive Mapping**: Visualizes property data on a map of Nairobi County (`nairobi_affordability_map.html`), allowing for geographic exploration of affordability.

##  Repository Structure

Here's a breakdown of the key files and directories in this repository:

| File/Directory | Description |
| :--- | :--- |
| **Data Files** |
| [`all_raw_listings.csv`](./all_raw_listings.csv) | The initial, unprocessed dataset scraped from the web. Contains raw text and prices. |
| [`cleaned_properties.csv`](./cleaned_properties.csv) | The dataset after running the cleaning and parsing scripts. Prices are normalized, and bedroom counts are extracted. |
| [`location_summary.csv`](./location_summary.csv) | A preliminary summary of properties grouped by location. |
| [`location_summary_clean.csv`](./location_summary_clean.csv) | The final, cleaned location summary with key statistics (e.g., average price, average bedrooms, count of listings). |
| [`nairobi_county.geojson`](./nairobi_county.geojson) | Geographic boundary file for Nairobi County, used for creating maps. |
| [`nairobi_affordability_map.html`](./nairobi_affordability_map.html) | An interactive HTML map visualizing property affordability across Nairobi. |
| [`nairobi_county_map.html`](./nairobi_county_map.html) | A base map of Nairobi County boundaries. |
| **Python Scripts** |
| [`scrape_listings.py`](./scrape_listings.py) | The web scraping script. It collects raw property data and saves it to `all_raw_listings.csv`. |
| [`nairobi_property_scraper_v2.py`](./nairobi_property_scraper_v2.py) | An updated or alternative version of the scraper. |
| [`clean_properties.py`](./clean_properties.py) | Cleans the raw data, handles missing values, and standardizes formats. |
| [`parser.py`](./parser.py) | Contains the core logic for extracting bedroom counts from titles and URLs. |
| [`prepare_properties.py`](./prepare_properties.py) | Orchestrates the cleaning, parsing, and enrichment process to create `cleaned_properties.csv`. |
| [`build_summary.py`](./build_summary.py) | Aggregates the cleaned data to create location-based summaries. |
| [`eda.py`](./eda.py) | A script for performing Exploratory Data Analysis, generating initial insights and statistics. |
| [`map_nairobi.py`](./map_nairobi.py) | Generates the interactive affordability map (`nairobi_affordability_map.html`). |
| [`audit_data.py`](./audit_data.py) | A utility script to check the quality and integrity of the data files. |
| **Documentation** |
| [`README.md`](./README.md) | This file. |
| [`data_dictionary.md`](./data_dictionary.md) | A detailed explanation of each column in the datasets. |
| **Requirements** |
| [`requirements.txt`](./requirements.txt) | A list of Python libraries required to run the project's scripts. |

## 🛠️ How to Use

Follow these steps to run the pipeline yourself.

### Prerequisites
*   Python 3.8 or higher installed on your system.
*   `pip` (Python package installer).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mainamuragev/nairobi_property_pricing.git
    cd nairobi_property_pricing
    ```

2.  **Install the required libraries:**
    It is highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

The typical workflow is sequential. You can run each script as you progress.

1.  **Scrape New Data (Optional):**
    If you want fresh data, run the scraper. This will overwrite `all_raw_listings.csv`.
    ```bash
    python scrape_listings.py
    ```
    *(Note: You may need to inspect the script and adapt the selectors if the target website structure has changed.)*

2.  **Clean and Prepare Data:**
    This step reads the raw data, applies cleaning and parsing, and produces the enriched dataset.
    ```bash
    python prepare_properties.py
    ```
    *   **Input:** `all_raw_listings.csv`
    *   **Output:** `cleaned_properties.csv`

3.  **Generate Location Summaries:**
    This script aggregates the cleaned data by location.
    ```bash
    python build_summary.py
    ```
    *   **Input:** `cleaned_properties.csv`
    *   **Output:** `location_summary_clean.csv`

4.  **Create the Affordability Map:**
    Generate an interactive HTML map to visualize the results.
    ```bash
    python map_nairobi.py
    ```
    *   **Inputs:** `cleaned_properties.csv`, `nairobi_county.geojson`
    *   **Output:** `nairobi_affordability_map.html`

5.  **Run Exploratory Analysis:**
    To see basic statistics and visualizations from the cleaned data.
    ```bash
    python eda.py
    ```

##  Data Dictionary

For a full description of all columns found in the CSV files (e.g., `price_per_bedroom`, `bedrooms`, `location`), please refer to the **[`data_dictionary.md`](./data_dictionary.md)** file.

##  Insights & Potential Use Cases

The final outputs can be used for:
*   **Market Analysis**: Identifying overpriced or undervalued neighborhoods.
*   **Investment Decisions**: Finding locations with the best rental yield based on price per bedroom.
*   **Personal House Hunting**: Using the interactive map to quickly find areas within a specific budget.
*   **Journalism & Research**: Supporting stories or studies on housing affordability in Nairobi.

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mainamuragev/nairobi_property_pricing/issues) if you have ideas on how to improve the parsing logic, add new data sources, or enhance the visualizations.

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Author

**mainamuragev**

*   GitHub: [@mainamuragev](https://github.com/mainamuragev)
