data_folder = 'data_copy'

import sqlite3
import pandas as pd
import os

# Connect to existing SQLite database (do not create or modify schema)
conn = sqlite3.connect('yelp_all_fromcsv.db')
cursor = conn.cursor()

data_folder = 'data_copy'

# Define the schema columns for each table (same as in step1_LoadAllDataTosqlDB.py)
business_cols = ['business_id', 'name', 'state', 'city', 'postal_code', 'latitude', 'longitude', 'stars', 'categories']
reviews_cols = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']
users_cols = [
    'user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool', 'elite', 'friends', 'fans',
    'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
    'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'
]

# Map CSV files to their corresponding table names and columns
csv_table_map = {
    'yelp_restaurants_filtered_with_new_features.csv': ('business', business_cols),
    'yelp_restaurant_reviews_with_vader.csv': ('reviews', reviews_cols),
    'yelp_users_with_taste_profile.csv': ('users', users_cols),
}

for csv_file, (table_name, schema_cols) in csv_table_map.items():
    csv_path = os.path.join(data_folder, csv_file)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Only keep columns that match the schema
        df_subset = df[[col for col in schema_cols if col in df.columns]]
        df_subset.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Inserted {len(df_subset)} rows into '{table_name}' from '{csv_file}'")
    else:
        print(f"File not found: {csv_path}")

conn.commit()

# Print row counts
for table_name, _ in csv_table_map.values():
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    print(f"{table_name.capitalize()} rows:", cursor.fetchone()[0])

conn.close()
