import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .timer import timeit


@timeit('Getting data')
def load_data(user_path, restaurant_path, reviews_path):
    """
    Load user, restaurant, and review data from CSV files.

    Args:
        user_path: Path to user features CSV file
        restaurant_path: Path to restaurant features CSV file
        reviews_path: Path to reviews CSV file

    Returns:
        tuple: (rdf, business_df, user_df) DataFrames
    """
    print("Loading data...")
    rdf = pd.read_csv(reviews_path)
    business_df = pd.read_csv(restaurant_path)
    user_df = pd.read_csv(user_path)
    print("Data loaded successfully.")
    return rdf, business_df, user_df


@timeit('Filtering review data')
def filter_review_data(rdf, min_user_reviews=20, min_rest_reviews=50):
    """
    Args:
        rdf: DataFrame containing review data
        min_user_reviews: Minimum number of reviews required per user
        min_rest_reviews: Minimum number of reviews required per restaurant

    Returns:
        DataFrame: Filtered reviews
    """
    print("Filtering data...")

    # filter businesses with at least min_rest_reviews
    business_review_counts = rdf['business_id'].value_counts()
    valid_businesses = business_review_counts[business_review_counts >= min_rest_reviews].index
    filtered_reviews = rdf[rdf['business_id'].isin(valid_businesses)]

    # filter users with at least min_user_reviews
    user_review_counts = filtered_reviews['user_id'].value_counts()
    valid_users = user_review_counts[user_review_counts >= min_user_reviews].index
    filtered_reviews = filtered_reviews[filtered_reviews['user_id'].isin(valid_users)]

    # apply business filter again to ensure consistency
    popular_businesses = business_review_counts[business_review_counts >= min_rest_reviews].index
    filtered_reviews = filtered_reviews[filtered_reviews['business_id'].isin(popular_businesses)]

    # print useful info
    print("Data filtered successfully.\n")
    print(f"After filtering (>={min_user_reviews} user reviews, >={min_rest_reviews} business reviews):")
    print(f"Users: {filtered_reviews['user_id'].nunique():,}")
    print(f"Businesses: {filtered_reviews['business_id'].nunique():,}")
    print(f"Reviews: {len(filtered_reviews):,}")

    # calculate sparsity
    sparsity = 1 - (len(filtered_reviews) / (filtered_reviews['user_id'].nunique() * filtered_reviews['business_id'].nunique()))
    print(f"Sparsity: {sparsity:.4f}")

    return filtered_reviews


def filter_business_data(business_df, filtered_reviews):
    """
    filter business data to include only businesses present in filtered reviews.

    Args:
        business_df: DataFrame containing business features
        filtered_reviews: DataFrame containing filtered review data

    Returns:
        DataFrame: Filtered business data
    """
    valid_business_ids = filtered_reviews['business_id'].unique()
    filtered_business_df = business_df[business_df['business_id'].isin(valid_business_ids)].copy()
    return filtered_business_df


def get_available_restaurant_columns(filtered_business_df):
    """
    Identify which restaurant feature columns are available in the dataset.

    Args:
        filtered_business_df: DataFrame containing business features

    Returns:
        tuple: (numeric_cols, categorical_cols, info_cols) lists of column names
    """
    print(f"\nFiltered businesses: {len(filtered_business_df):,}")

    # define numeric cols for business features
    business_numeric_cols = [
        'latitude', 'longitude', 'stars', 'review_count',
        'num_tips', 'num_checkins', 'RestaurantsPriceRange2',
        'avg_sentiment', 'local_preference_ratio', 'elite_user_ratio'
    ]

    # define categorical cols for business features
    business_categorical_cols = [
        'city', 'state', 'is_open',
        'has_coffee_or_tea', 'serves_sweets',
        'serves_alcohol', 'DogsAllowed', 'has_european_food',
        'has_american', 'has_asian_food', 'has_seafood',
        'has_entertainment', 'has_hispanic_food',
        'has_vegetarian', 'Open24Hours'
    ]

    # info cols to preserve (not encode)
    business_info_cols = ['business_id', 'name', 'address', 'categories']

    # check which cols actually exist
    available_numeric = [col for col in business_numeric_cols if col in filtered_business_df.columns]
    available_categorical = [col for col in business_categorical_cols if col in filtered_business_df.columns]
    available_info = [col for col in business_info_cols if col in filtered_business_df.columns]

    return available_numeric, available_categorical, available_info


@timeit('Preprocessing business features')
def preprocess_business_features(filtered_business_df, available_numeric, available_categorical):
    """
    Preprocess business features by scaling numeric columns and encoding categorical ones.

    Args:
        filtered_business_df: DataFrame containing business data
        available_numeric: List of numeric column names
        available_categorical: List of categorical column names

    Returns:
        DataFrame: Preprocessed business features with business_id preserved
    """
    # extract business features - keep original business_id separate
    business_features = filtered_business_df[['business_id'] + available_numeric + available_categorical].copy()

    # fill missing values
    business_features[available_numeric] = business_features[available_numeric].fillna(0)
    business_features = business_features.fillna(0)

    # scale numeric business columns
    scaler_business = StandardScaler()
    business_features[available_numeric] = scaler_business.fit_transform(
        business_features[available_numeric])

    # encode categorical variables using label encoding
    for col in available_categorical:
        if col in business_features.columns:
            business_features[col] = business_features[col].astype('category').cat.codes

    print("Any NaNs in business_features:", business_features.isna().any().any())
    return business_features


@timeit('Preprocessing user features')
def preprocess_user_features(filtered_reviews, user_df):
    """
    Preprocess user features by filtering active users, scaling, and encoding.

    Args:
        filtered_reviews: DataFrame containing filtered review data
        user_df: DataFrame containing user features

    Returns:
        DataFrame: Preprocessed user features with user_id preserved
    """
    # filter user_df to only include users in filtered_reviews
    active_user_ids = filtered_reviews['user_id'].unique()
    user_df_filtered = user_df[user_df['user_id'].isin(active_user_ids)].copy()

    # categorical columns that need encoding
    user_categorical_cols = ['base_city', 'state_code', 'top_category', 'elite']

    # Define user numeric feature columns
    user_numeric_cols = [
        'review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars',
        'compliment_hot', 'compliment_more', 'compliment_profile',
        'compliment_cute', 'compliment_list', 'compliment_note',
        'compliment_plain', 'compliment_cool', 'compliment_funny',
        'compliment_writer', 'compliment_photos', 'avg_sentiment',
        'top_category_ratio', 'has_american_ratio', 'has_asian_food_ratio',
        'has_hispanic_food_ratio', 'has_european_food_ratio', 'has_seafood_ratio',
        'serves_alcohol_ratio', 'has_vegetarian_ratio', 'has_entertainment_ratio',
        'has_coffee_or_tea_ratio', 'serves_sweets_ratio', 'years_elite',
        'rating_variance', 'local_review_ratio', 'days_since_last_review',
        'review_frequency', 'avg_price_preference', 'num_cities_reviewed']

    # prep user features with encoding
    user_features_prepared = user_df_filtered[['user_id']].copy()

    # add numeric features directly
    available_user_numeric = [col for col in user_numeric_cols if col in user_df.columns]
    for col in available_user_numeric:
        user_features_prepared[col] = user_df_filtered[col].fillna(0)

    # scale numeric columns
    scaler_user = StandardScaler()
    user_features_prepared[available_user_numeric] = scaler_user.fit_transform(
        user_features_prepared[available_user_numeric])

    # encode cat features
    available_user_categorical = [col for col in user_categorical_cols if col in user_df.columns]
    for col in available_user_categorical:
        user_features_prepared[f'{col}_encoded'] = user_df_filtered[col].astype('category').cat.codes

    # handle date feature (convert to days since first review)
    if 'yelping_since' in user_df.columns:
        user_df_temp = user_df_filtered.copy()
        user_df_temp['yelping_since'] = pd.to_datetime(user_df_temp['yelping_since'], errors='coerce')
        reference_date = user_df_temp['yelping_since'].max()
        days_yelping = (reference_date - user_df_temp['yelping_since']).dt.days
        user_features_prepared['days_yelping'] = days_yelping.fillna(0)

        # scale days_yelping
        user_features_prepared['days_yelping'] = scaler_user.fit_transform(
            user_features_prepared['days_yelping'].values.reshape(-1, 1))

    # final cleanup
    user_features_prepared = user_features_prepared.fillna(0)

    # add prints for debugging
    print(f"User features shape: {user_features_prepared.shape}")
    print("Any NaNs in user_features_prepared:", user_features_prepared.isna().any().any())
    print(f"Columns: {user_features_prepared.columns.tolist()}")
    return user_features_prepared

