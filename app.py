import streamlit as st
import pandas as pd

# --- Load the dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv", engine='python', on_bad_lines='skip', encoding='utf-8')
    df = df[['name', 'location', 'cuisines', 'approx_cost(for two people)', 
             'online_order', 'book_table', 'rate', 'votes']]
    df.dropna(inplace=True)

    # Clean 'rate' column
    def clean_rate(x):
        try:
            if '/' in str(x):
                return float(x.split('/')[0])
            else:
                return float(x)
        except:
            return 0
    df['rate'] = df['rate'].apply(clean_rate)

    # Clean 'approx_cost' column
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].replace(',', '', regex=True)
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    df.dropna(subset=['approx_cost(for two people)'], inplace=True)

    # Normalize strings
    df['location'] = df['location'].str.strip().str.lower()
    df['cuisines'] = df['cuisines'].str.strip().str.lower()

    # Convert Yes/No to 0/1
    df['online_order'] = df['online_order'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['book_table'] = df['book_table'].apply(lambda x: 1 if x == 'Yes' else 0)

    return df

df = load_data()

# --- Recommendation function ---
def recommend_restaurants(location, cuisine, budget_min, budget_max, 
                          online_order=None, book_table=None, min_rating=0):
    temp_df = df[df['location'].str.contains(location, na=False)]
    temp_df = temp_df[temp_df['cuisines'].str.contains(cuisine, na=False)]
    temp_df = temp_df[(temp_df['approx_cost(for two people)'] >= budget_min) & 
                      (temp_df['approx_cost(for two people)'] <= budget_max)]
    
    if online_order is not None:
        temp_df = temp_df[temp_df['online_order'] == online_order]
    if book_table is not None:
        temp_df = temp_df[temp_df['book_table'] == book_table]
    
    temp_df = temp_df[temp_df['rate'] >= min_rating]
    
    temp_df = temp_df.sort_values(by=['rate', 'votes'], ascending=False)
    
    return temp_df[['name', 'location', 'cuisines', 'approx_cost(for two people)', 
                    'rate', 'online_order', 'book_table']].head(10)

# --- Streamlit UI ---
st.title("🍽️ Zomato Bangalore Restaurant Recommender")

# User Inputs
location = st.text_input("Enter Location (e.g., Indiranagar):").strip().lower()
cuisine = st.text_input("Enter Cuisine (e.g., Italian):").strip().lower()
budget = st.slider("Select Budget Range (for two people)", 100, 5000, (300, 1500))
min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
online_order = st.radio("Online Ordering?", ("Doesn't Matter", "Yes", "No"))
book_table = st.radio("Table Booking?", ("Doesn't Matter", "Yes", "No"))

# Convert inputs
online_order_val = 1 if online_order == "Yes" else 0 if online_order == "No" else None
book_table_val = 1 if book_table == "Yes" else 0 if book_table == "No" else None

# Recommend button
if st.button("🔍 Recommend Restaurants"):
    if location and cuisine:
        results = recommend_restaurants(location, cuisine,
                                        budget[0], budget[1],
                                        online_order_val, book_table_val,
                                        min_rating)
        if results.empty:
            st.warning("No restaurants found matching your preferences. Try adjusting filters.")
        else:
            st.success("Here are the top recommendations:")
            st.dataframe(results)
    else:
        st.error("Please enter both location and cuisine.")
