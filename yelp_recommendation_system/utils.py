import dill
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def save_model(model, filename='hybrid_recommender_model.pkl'):
    """
    Save a trained model to disk using dill for serialization.

    Args:
        model: Trained HybridRecommender instance
        filename: Path where the model should be saved
    """
    with open(filename, 'wb') as f:
        dill.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename='hybrid_recommender_model.pkl'):
    """
    Load a trained model from disk.

    Args:
        filename: Path to the saved model file

    Returns:
        Loaded HybridRecommender instance
    """
    with open(filename, 'rb') as f:
        model = dill.load(f)
    print(f"Model loaded from {filename}")
    return model


def get_user_recommendations(
    model,
    user_id: str,
    business_features: pd.DataFrame,
    reviews: pd.DataFrame,
    business_df: pd.DataFrame,
    n: int = 10,
    min_rating: float = 3.5) -> pd.DataFrame:
    """
    Get top N restaurant recommendations for a specific user.

    This function generates predictions for all restaurants the user hasn't
    reviewed yet, filters by minimum rating, and returns the top N.

    Args:
        model: Trained HybridRecommender instance
        user_id: User identifier
        business_features: DataFrame containing business feature data
        reviews: DataFrame containing review data
        business_df: DataFrame containing business metadata
        n: Number of recommendations to return
        min_rating: Minimum predicted rating threshold

    Returns:
        DataFrame with top N recommendations including business details
    """
    # get all businesses
    all_businesses = reviews['business_id'].unique()

    # get businesses the user has already rated
    user_rated = reviews[reviews['user_id'] == user_id]['business_id'].unique()

    # get unrated businesses
    unrated_businesses = [b for b in all_businesses if b not in user_rated]

    # filter
    trainset_items = [model.trainset.to_raw_iid(i) for i in range(model.trainset.n_items)]
    unrated_in_trainset = [b for b in unrated_businesses if b in trainset_items]
    predictions = []

    for business_id in unrated_in_trainset:
        try:
            # convert to internal ids
            inner_uid = model.trainset.to_inner_uid(user_id)
            inner_iid = model.trainset.to_inner_iid(business_id)

            # get prediction
            pred_rating = model.estimate(inner_uid, inner_iid)
            predictions.append({'user_id':user_id, 'business_id': business_id, 'predicted_rating': pred_rating})

        except:
            # user or business not in training set
            # print(f"Skipping prediction for user {user_id} and business {business_id}")
            continue
    # print("pred: ", predictions)


    # convert to df and filter
    recs_df = pd.DataFrame(predictions)

    if len(recs_df) == 0:
        print(f"No recommendations available for user {user_id}")
        return pd.DataFrame()

    recs_df = recs_df[recs_df['predicted_rating'] >= min_rating]
    recs_df = recs_df.sort_values('predicted_rating', ascending=False).head(n)

    # add business details
    merge_cols = ['business_id', 'name', 'city', 'state', 'stars', 'categories']
    available_cols = ['business_id'] + [col for col in merge_cols[1:] if col in business_df.columns]

    recs_df = recs_df.merge(business_df[available_cols], on='business_id', how='left')
    return recs_df


def get_batch_predictions(
    model,
    user_business_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Get preds for multiple user-business pairs.

    Args:
        model: HybridRecommender instance
        user_business_pairs: list of (user_id, business_id) tuples

    Returns:
        dataframe with columns: user_id, business_id, predicted_rating
    """
    results = []

    for user_id, business_id in user_business_pairs:
        try:
            inner_uid = model.trainset.to_inner_uid(user_id)
            inner_iid = model.trainset.to_inner_iid(business_id)
            pred_rating = model.estimate(inner_uid, inner_iid)

            results.append({
                'user_id': user_id,
                'business_id': business_id,
                'predicted_rating': pred_rating})

        except:
            # user or business not in training set
            results.append({
                'user_id': user_id,
                'business_id': business_id,
                'predicted_rating': None})

    return pd.DataFrame(results)


def create_train_test_sets(filtered_reviews, test_size=0.2, random_state=23):
    """
    Create train and test sets from filtered reviews using Surprise.

    Args:
        filtered_reviews: DataFrame containing filtered review data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (trainset, testset) Surprise dataset objects
    """
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split

    # create working dataset with essential columns
    reviews_clean = filtered_reviews[['user_id', 'business_id', 'stars']].copy()

    # check for ratings distribution
    print("Ratings distribution:")
    print(reviews_clean['stars'].value_counts().sort_index(), "\n")
    print("Creating train and test sets...\n")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(reviews_clean[['user_id', 'business_id', 'stars']], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    print("Train and test sets created successfully.\n")
    return trainset, testset


def build_feat_vector_for_new_user(new_user_dict):
    """
    Build a feature vector for a new user
    """
    USER_FEAT_COLS = [
        'review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
        'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer','compliment_photos', 'avg_sentiment', 'top_category_ratio',  'has_american_ratio', 'has_asian_food_ratio', 'has_hispanic_food_ratio',
        'has_european_food_ratio', 'has_seafood_ratio', 'serves_alcohol_ratio', 'has_vegetarian_ratio',
        'has_entertainment_ratio', 'has_coffee_or_tea_ratio', 'serves_sweets_ratio', 'years_elite',
        'rating_variance', 'local_review_ratio', 'days_since_last_review', 'review_frequency', 'avg_price_preference', 'num_cities_reviewed', 'base_city_encoded',
        'state_code_encoded', 'top_category_encoded', 'elite_encoded', 'days_yelping']

    USER_FLOAT_COLS = [
        'review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile',
        'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos',
        'avg_sentiment', 'top_category_ratio', 'has_american_ratio', 'has_asian_food_ratio',
        'has_hispanic_food_ratio', 'has_european_food_ratio', 'has_seafood_ratio', 'serves_alcohol_ratio',
        'has_vegetarian_ratio', 'has_entertainment_ratio', 'has_coffee_or_tea_ratio',
        'serves_sweets_ratio', 'years_elite', 'rating_variance', 'local_review_ratio', 'days_since_last_review', 'review_frequency',
        'avg_price_preference', 'num_cities_reviewed', 'days_yelping']
    user_scaler = StandardScaler()
    # fill missing features with 0
    vector_values = []
    for col in USER_FEAT_COLS:
        vector_values.append(new_user_dict.get(col, 0.0))

    df = pd.DataFrame([vector_values], columns=USER_FEAT_COLS)

    # scale float columns if scaler exists
    # if user_scaler is not None:
    # df[USER_FLOAT_COLS] = user_scaler.transform(df[USER_FLOAT_COLS])

    # convert to numpy
    return df.to_numpy(dtype=np.float32).ravel()


def recommend_for_new_user(model, user_feat_vector, top_n=100):
    """
    Recommend items for a brand-new user using only their features.

    Args:
        user_feat_vector: 1D numpy array of user featurs
        top_n: number of items to return

    Returns:
        List of (raw_item_id, predicted_rating) sorted by score desc.
    """
    # bild latent user vector from features
    user_feat_vector = np.nan_to_num(user_feat_vector, nan=0.0)
    user_latent = user_feat_vector @ model.Wu

    scores = []
    for inner_iid in range(model.trainset.n_items):
        # item bias + feature/latent interactions
        pred = model.global_mean + model.bi[inner_iid]

        # user-features with business latent
        pred += model.alpha * np.dot(user_latent, model.qi[inner_iid])

        # business features branch
        if model.has_loc_features and model.beta > 0.0:
            loc_latent = model.loc_feat_matrix[inner_iid] @ model.Wi
            pred += model.beta * np.dot(user_latent, loc_latent)

        raw_iid = model.trainset.to_raw_iid(inner_iid)
        scores.append((raw_iid, pred))

    # sort by predicted score and keep top_n
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]