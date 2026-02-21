import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(page_title="Nairobi Property Dashboard", layout="wide")
st.title("📊 Nairobi Property Market Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_properties.csv')
    
    # Try to parse month column intelligently
    # First, check if month column exists and is not all null
    if 'month' in df.columns and df['month'].notna().any():
        # Attempt common formats: 'YYYY-MM', 'Month YYYY', 'YYYY-M', etc.
        # Use infer with errors='coerce' to handle various formats
        df['month_dt'] = pd.to_datetime(df['month'], errors='coerce', infer_datetime_format=True)
        # If that fails, try a custom format (e.g., 'Feb 2026')
        if df['month_dt'].isna().all():
            df['month_dt'] = pd.to_datetime(df['month'], format='%b %Y', errors='coerce')
    else:
        df['month_dt'] = pd.NaT  # create empty column
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
all_locations = sorted(df['location'].dropna().unique())
selected_locations = st.sidebar.multiselect(
    "Locations",
    options=all_locations,
    default=[]
)
min_bed, max_bed = st.sidebar.slider(
    "Bedrooms",
    min_value=int(df['bedrooms'].min()) if not df['bedrooms'].isna().all() else 1,
    max_value=int(df['bedrooms'].max()) if not df['bedrooms'].isna().all() else 5,
    value=(1, 5)
)

# Apply filters
filtered_df = df.copy()
if selected_locations:
    filtered_df = filtered_df[filtered_df['location'].isin(selected_locations)]
filtered_df = filtered_df[(filtered_df['bedrooms'] >= min_bed) & (filtered_df['bedrooms'] <= max_bed)]

# Helper function to truncate long location names
def truncate(name, max_len=30):
    return name if len(name) <= max_len else name[:max_len] + '...'

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Median Price by Location")
    # Top 15 locations by median price
    if not filtered_df.empty and 'price_normalized' in filtered_df.columns:
        loc_median = filtered_df.groupby('location')['price_normalized'].median().sort_values(ascending=False).head(15)
        if not loc_median.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Truncate labels
            short_labels = [truncate(loc) for loc in loc_median.index]
            ax.barh(short_labels, loc_median.values, color='skyblue')
            ax.set_xlabel("Median Price (KES)")
            ax.set_ylabel("")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No data for selected filters.")
    else:
        st.info("Price data not available.")

with col2:
    st.subheader("📈 Monthly Price Trend")
    # Group by month and calculate median price
    if 'month_dt' in filtered_df.columns and filtered_df['month_dt'].notna().any():
        monthly = filtered_df.groupby('month_dt')['price_normalized'].median().reset_index()
        monthly = monthly.dropna()
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(monthly['month_dt'], monthly['price_normalized'], marker='o', linestyle='-', color='coral')
            ax.set_xlabel("Month")
            ax.set_ylabel("Median Price (KES)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No monthly data available for selected filters.")
    else:
        st.info("Month data not available or all null.")

# Second row
col3, col4 = st.columns(2)

with col3:
    st.subheader("🏷️ Price per Bedroom (Top Affordable Areas)")
    if 'price_per_bedroom' in filtered_df.columns:
        ppb = filtered_df.groupby('location')['price_per_bedroom'].mean().sort_values().head(10)
        if not ppb.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            short_labels = [truncate(loc) for loc in ppb.index]
            ax.barh(short_labels, ppb.values, color='lightgreen')
            ax.set_xlabel("Avg Price per Bedroom (KES)")
            ax.set_ylabel("")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No data for selected filters.")
    else:
        st.info("Price per bedroom column not found.")

with col4:
    st.subheader("🏢 Market Segments by Bedrooms")
    # Count of listings per bedroom number
    if 'bedrooms' in filtered_df.columns:
        bed_counts = filtered_df['bedrooms'].value_counts().sort_index()
        if not bed_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            bed_counts.plot(kind='bar', ax=ax, color='purple')
            ax.set_xlabel("Bedrooms")
            ax.set_ylabel("Number of Listings")
            # Rotate x labels for readability
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No data for selected filters.")
    else:
        st.info("Bedrooms column not found.")

# Data table at bottom
st.subheader("🔍 Filtered Data Preview")
if not filtered_df.empty:
    display_cols = ['location', 'bedrooms', 'price_normalized', 'price_per_bedroom', 'month']
    # Only show columns that exist
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[display_cols].head(100))
else:
    st.info("No data matching filters.")

st.caption("Source: Cleaned Nairobi property listings")
