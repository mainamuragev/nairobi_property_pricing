# Nairobi Property Pricing Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url)  <!-- Replace with deployed URL -->

A complete data pipeline for scraping, cleaning, enriching, and **predicting** property prices in Nairobi. This project extends the original analysis with machine learning models and interactive tools for investors, tenants, and analysts.

## 🚀 New Features

- **Machine Learning Price Prediction**: Trained a Random Forest model (MAE = **KES 4.8M**) to estimate property prices based on location and bedrooms.
- **Interactive Web App**: [`app.py`](app.py) – Enter details and get an estimated price with a realistic range.
- **Market Dashboard**: [`dashboard.py`](dashboard.py) – Filterable charts for median prices, monthly trends, affordability, and market segments.
- **Feature Importance Analysis**: Identified top price drivers – bedroom count and premium locations like Westlands, Loresho, and Kiambu Road.

## 📊 Modeling Performance

| Model | MAE (KES) | RMSE (KES) | R² |
|-------|-----------|------------|-----|
| Linear Regression | 5,669,566 | 8,339,069 | 0.240 |
| Random Forest | **4,801,158** | **8,011,117** | **0.299** |

**Random Forest improved MAE by 15.3%**.

![Feature Importance](feature_importance_top5.png)
*Top 5 price drivers from the Random Forest model.*

## 🖥️ Interactive Apps

After installing dependencies (`pip install -r requirements.txt`), run:

**Price Predictor**
```bash
streamlit run app.py
```
**Market Dashboard**
```bash
streamlit run dashboard.py
```

## 📦 Updated Repository Structure

| File/Directory | Description |
| :--- | :--- |
| **Original Pipeline** | |
| `scrape_listings.py` | Web scraper for raw listings |
| `clean_properties.py` | Cleans and normalizes data |
| `parser.py` | Extracts bedroom counts from text |
| `prepare_properties.py` | Orchestrates cleaning & enrichment |
| `build_summary.py` | Generates location summaries |
| `map_nairobi.py` | Creates interactive affordability map |
| `eda.py` | Exploratory analysis & charts |
| `audit_data.py` | Data quality checks |
| **New Modeling & Apps** | |
| `train_random_forest.py` | Trains and evaluates Random Forest model |
| `model.pkl` | Saved trained model |
| `app.py` | Streamlit price prediction app |
| `dashboard.py` | Streamlit market dashboard |
| `feature_importance_top5.png` | Plot of top price drivers |
| **Data Files** | |
| `all_raw_listings.csv` | Raw scraped data |
| `cleaned_properties.csv` | Cleaned dataset with price_normalized |
| `location_summary_clean.csv` | Aggregated location stats |
| `nairobi_affordability_map.html` | Interactive map |
| ... (other data files remain) |

## 📈 Dashboard Preview

*(Add screenshots of your dashboard here if desired)*

## 🔮 Next Steps

- Deploy apps online via Streamlit Cloud.
- Incorporate additional features (property size, amenities) for better accuracy.
- Add time-series forecasting for market trends.

---

*The rest of the original README follows:*

## Overview

This project automates the collection and processing of Nairobi real estate data. It is designed to answer key questions like: "What is the average rent per bedroom in different areas?" and "Where are the most affordable neighborhoods based on my budget?".

The pipeline consists of several stages:
1.  **Scraping**: Collecting raw listing data from property websites.
2.  **Cleaning & Parsing**: Normalizing prices and intelligently extracting the number of bedrooms from messy titles and URLs.
3.  **Enrichment**: Calculating affordability metrics like price per bedroom.
4.  **Analysis & Visualization**: Generating location-based summaries and an interactive map.

## Key Features

*   **Automated Data Pipeline**: A series of Python scripts that handle the entire data workflow from `scrape` to `analyze`.
*   **Intelligent Bedroom Parsing**: Extracts bedroom counts from unstructured text (e.g., "2br", "2 bedroom", "2 bed") using the `parser.py` module.
*   **Price Normalization**: Cleans and standardizes price formats (e.g., "Ksh 45,000", "45k") into numerical values for analysis.
*   **Affordability Calculation**: Computes the **price per bedroom**, a key metric for comparing value across different property sizes.
*   **Location Summaries**: Aggregates data by area to generate summary statistics like average rent and bedroom counts (`location_summary_clean.csv`).
*   **Interactive Mapping**: Visualizes property data on a map of Nairobi County (`nairobi_affordability_map.html`), allowing for geographic exploration of affordability.

## Visualizations

### Interactive Affordability Map
![Affordability Map](https://github.com/mainamuragev/nairobi_property_pricing/raw/main/docs/images/affordability_map.png)
*Figure 1: Map showing property affordability across Nairobi. Open [`nairobi_affordability_map.html`](nairobi_affordability_map.html) for the interactive version.*

### Exploratory Data Analysis
**Price Distribution**
![Price Distribution](https://github.com/mainamuragev/nairobi_property_pricing/raw/main/docs/images/price_distribution.png)
*Figure 2: Histogram of rental prices.*

**Bedrooms vs. Price**
![Bedrooms vs Price](https://github.com/mainamuragev/nairobi_property_pricing/raw/main/docs/images/price_by_bedrooms.png)
*Figure 3: Scatter plot of price by number of bedrooms.*

**Market Segments**
![Tier Breakdown](https://github.com/mainamuragev/nairobi_property_pricing/raw/main/docs/images/tier_breakdown.png)
*Figure 4: Proportion of affordable, mid-range, and premium locations.*

### Full EDA Gallery
Click on any thumbnail to view the full-size chart.

| | | |
|:---:|:---:|:---:|
| [![Top/Bottom Prices](charts/01_top_bottom_prices.png)](charts/01_top_bottom_prices.png) | [![Price Distribution](charts/02_price_distribution.png)](charts/02_price_distribution.png) | [![Price by Bedrooms](charts/03_price_by_bedrooms.png)](charts/03_price_by_bedrooms.png) |
| [![Price per Bedroom](charts/04_price_per_bedroom.png)](charts/04_price_per_bedroom.png) | [![Rank vs Price](charts/05_rank_vs_price_scatter.png)](charts/05_rank_vs_price_scatter.png) | [![Avg vs Median Gap](charts/06_avg_vs_median_gap.png)](charts/06_avg_vs_median_gap.png) |
| [![Tier Breakdown](charts/07_tier_breakdown.png)](charts/07_tier_breakdown.png) | [![Correlation Heatmap](charts/08_correlation_heatmap.png)](charts/08_correlation_heatmap.png) | [![Full Price Ranking](charts/09_full_price_ranking.png)](charts/09_full_price_ranking.png) |

## How to Use

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
    *   **Output:** Nine charts saved in the `charts/` folder.

### Running the New Apps

After completing the pipeline, you can launch the interactive tools:

**Price Predictor**
```bash
streamlit run app.py
```

**Market Dashboard**
```bash
streamlit run dashboard.py
```

## Data Dictionary

For a full description of all columns found in the CSV files (e.g., `price_per_bedroom`, `bedrooms`, `location`), please refer to the **[`data_dictionary.md`](./data_dictionary.md)** file.

## Insights & Potential Use Cases

The final outputs can be used for:
*   **Market Analysis**: Identifying overpriced or undervalued neighborhoods.
*   **Investment Decisions**: Finding locations with the best rental yield based on price per bedroom.
*   **Personal House Hunting**: Using the interactive map to quickly find areas within a specific budget.
*   **Journalism & Research**: Supporting stories or studies on housing affordability in Nairobi.

## Contributing

Contributions, issues, and feature requests are welcome! If you have ideas on how to improve the parsing logic, add new data sources, or enhance the visualizations, feel free to open an issue or pull request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**mainamuragev**
- GitHub: [@mainamuragev](https://github.com/mainamuragev)
```



