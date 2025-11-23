import pandas as pd
import json
import os

#data for recommendation comes in csv format and hidden_gem_score is in json format
# --- 1. Load Data ---

# Load the CSV file provided by the user
df_csv = pd.read_csv("./newdata/user_top25_recs.csv")
with open('./newdata/business_metrics.json', 'r') as f:
    json_data = json.load(f)


# 2. Process JSON and Select Target Column
df_json = pd.DataFrame(json_data)

# Select only the columns needed for the merge
df_gem_scores = df_json[['business_id', 'hidden_gem_score']]

# 3. Merge DataFrames
# Perform a left merge to keep all rows from the original CSV 
# and add the hidden_gem_score where a match is found.
df_merged = df_csv.merge(
    df_gem_scores,
    on='business_id',
    how='left'  # Use 'left' join to preserve all original recommendation rows
)

# 4. Save the Result to a new CSV file
output_filename = "user_recs_25_with_gem_score.csv"
df_merged.to_csv(output_filename, index=False)

print(f"Successfully merged data. The 'hidden_gem_score' column has been added to the CSV.")
print(f"Resulting file saved as: {output_filename}")