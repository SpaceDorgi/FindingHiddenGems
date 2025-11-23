from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import json

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# write in function for .py use later
def determine_base_city(user_reviews, businesses):
    """
    Determine a user's base city from their review history.

    Args:
        user_reviews: List of review dicts for a specific user
        businesses: Dict mapping business_id to business dict

    Returns:
        String representing the user's base city, or None if no reviews
    """
    cities = []

    for review in user_reviews:
        business_id = review['business_id']
        if business_id in businesses:
            city = businesses[business_id]['city']
            cities.append(city)

    if not cities:
        return None

    # return the most common city
    city_counts = Counter(cities)
    base_city, _ = city_counts.most_common(1)[0]

    return base_city

def classify_review(business, base_city):
    """
    Classify a review as 'local' or 'tourist'.

    Args:
        review: Review dict
        business: Business dict
        base_city: User's base city string

    Returns:
        'local' if review is in base city, 'tourist' otherwise
    """
    business_city = business['city']
    return 'local' if business_city == base_city else 'tourist'

# Example usage
def analyze_user_reviews(user_id, reviews, businesses):
    """
    Analyze all reviews for a user.

    Returns:
        Dict with base_city and classified reviews
    """
    user_reviews = [r for r in reviews if r['user_id'] == user_id]
    base_city = determine_base_city(user_reviews, businesses)

    classified_reviews = []
    for review in user_reviews:
        business = businesses.get(review['business_id'])
        if business:
            classification = classify_review(business, base_city)
            classified_reviews.append({'review_id': review['review_id'], 'business_city': business['city'], 'classification': classification})

    return {'user_id': user_id, 'base_city': base_city, 'reviews': classified_reviews}


def classify_all_reviews(reviews, businesses):
    """
    Classify all reviews as local or tourist for each user.

    Args:
        reviews: List of all review dicts
        businesses: List of all business dicts

    Returns:
        Dict mapping user_id to their analysis results
    """
    # greate business lookup dict for faster access
    business_dict = {b['business_id']: b for b in businesses}

    # group reviews by user
    user_reviews_map = {}
    for review in reviews:
        user_id = review['user_id']
        if user_id not in user_reviews_map:
            user_reviews_map[user_id] = []
        user_reviews_map[user_id].append(review)

    # analyze each user
    results = {}
    for user_id, user_reviews in user_reviews_map.items():
        results[user_id] = analyze_user_reviews(user_id, user_reviews, business_dict)

    return results


def add_base_city_to_users(users, all_results):
    """
    Add base_city field to each user in the users list.

    Args:
        users: List of user dicts
        all_results: Dict mapping user_id to analysis results

    Returns:
        Updated users list with base_city added
    """
    for user in users:
        user_id = user['user_id']
        # since user id is a key in the results dict, we can directly check and assign
        if user_id in all_results:
            user['base_city'] = all_results[user_id]['base_city']
        else:
            user['base_city'] = None

    return users


def add_classification_to_reviews(reviews, all_results):
    """
    Add location_classification field to each review.

    Args:
        reviews: List of review dicts
        all_results: Dict mapping user_id to analysis results

    Returns:
        Updated reviews list with location_classification added
    """
    # create lookup dict for faster access
    classification_lookup = {}
    for user_id, user_data in all_results.items():
        for review_data in user_data['reviews']:
            review_id = review_data['review_id']
            classification_lookup[review_id] = review_data['classification']

    # add classification to each review
    for review in reviews:
        review_id = review['review_id']
        review['location_classification'] = classification_lookup.get(review_id, None)

    return reviews



def build_city_state_map(restaurants: List[dict]) -> Dict[str, str]:
    """
    Return a mapping normalized_city -> most_common_state_code
    """
    counts = defaultdict(Counter)
    for r in restaurants:
        city = r.get('city')
        state = r.get('state') or r.get('state_code')
        if not city or not state:
            continue
        key = city.strip().lower()
        counts[key][state.strip()] += 1
    return {city: counter.most_common(1)[0][0] for city, counter in counts.items()}


def infer_user_states(users: List[dict], restaurants: Optional[List[dict]] = None,
                    restaurants_filepath: str = '../yelp_dataset/yelp_restaurants.json',
                    overwrite_state_code: bool = False) -> Tuple[List[dict], Dict[str,str]]:
    """
    For each user with a `base_city`, set `user['inferred_state']` and populate
    `user['state_code']` when missing (unless overwrite_state_code is True).
    Returns (updated_users, city_state_map).
    """
    if restaurants is None:
        with open(restaurants_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            restaurants = data.get('restaurants', data)  # support either shape

    city_state_map = build_city_state_map(restaurants)

    for u in users:
        base_city = u.get('base_city')
        if not base_city:
            continue
        key = base_city.strip().lower()
        state = city_state_map.get(key)
        if state:
            u['inferred_state'] = state
            if overwrite_state_code or not u.get('state_code'):
                u['state_code'] = state

    return users, city_state_map


# remove junk categories that may be in the data
def get_categories_to_remove() -> List[str]:
    """
    Returns a list of business categories to remove from consideration.
    """
    categories_to_remove = [
        'automotive', 'gas stations', 'fashion', 'hotels & travel', 'cosmetics & beauty supply', 'home & garden',
        'photography stores & services', 'electronics', 'furniture stores', 'tobacco shops', 'education',
        'home services', 'mobile phones', 'home decor', 'bookstores', 'appliances', 'hardware stores', 'vape shops',
        'head shops', 'religious organizations' 'party supplies', 'public services & government', 'medical centers',
        'community service/non-profit', 'tires', 'sporting goods', 'eyewear & opticians', 'hobby shops', 'nurseries & gardening', 'fitness & instruction', 'real estate', 'financial services', 'lawyers', 'insurance', 'shipping centers', 'professional services', 'shopping centers', 'food court', 'wholesale stores', 'dance clubs', 'hookah bars', 'massage' 'local services', 'international grocery', 'farmers market', 'department stores', 'beauty & spas', 'health & medical', 'convenience stores',]
    return categories_to_remove


# used to help create more features downstrea,
def create_common_mappings():
    category_mappings = {
        'has_american': ['american (traditional)', 'american (new)', 'diners', 'steakhouses', 'hot dogs', 'comfort food', 'southern', 'bbq', 'new american', 'breakfast & brunch', 'buffets', 'chicken wings', 'cajun/creole', 'gastropubs', 'meat shops', 'food stands', 'food trucks'],

        'has_asian_food': ['japanese', 'chinese', 'sushi bars', 'asian fusion', 'korean', 'indian', 'vietnamese', 'thai', 'laotian', 'malaysian',
                           'mongolian', 'taiwanese', 'cantonese', 'dim sum', 'hot pot', 'filipino', 'singaporean'
                           'szechuan', 'ramen'],
        'has_hispanic_food': ['mexican', 'tex-mex', 'spanish', 'puerto rican', 'cuban', 'colombian', 'salvadoran', 'argentine', 'peruvian', 'brazilian', 'venezuelan', 'dominican', 'guatemalan', 'honduran', 'empanadas' 'latin american', 'new mexican cuisine'],

        'has_european_food': ['italian', 'french', 'german', 'greek', 'mediterranean', 'spanish', 'portuguese', 'austrian', 'belgian', 'hungarian', 'irish', 'scandinavian', 'swiss', 'russian', 'polish', 'pasta', 'modern european'],

        'has_seafood': ['seafood', 'seafood markets', 'fish & chips', 'poke'],
        'serves_alcohol': ['beer, wine & spirits', 'pubs', 'cocktail bars', 'wine bars', 'beer bar', 'breweries', 'wineries', 'brewpubs'
                           'beer tours', 'wine tours', 'distilleries', 'whiskey bars', 'wine tasting classes', 'irish pub',
                           ],
        'has_vegetarian': ['vegan', 'vegetarian'],
        'has_entertainment': ['arts & entertainment', 'music venues', 'karaoke', 'dinner theatre', 'piano bars', ],
        'has_coffee_or_tea': ['coffee & tea', 'bubble tea', 'tea rooms', 'coffee roasteries', 'tea rooms'],
        'serves_sweets': ['desserts', 'ice cream & frozen yogurt', 'bakeries', 'donuts', 'cupcakes', 'chocolatiers & shops', 'shaved_ice', 'gelato', 'creperies', 'patisserie/cake shop', 'candy stores', 'custom cakes'],
    }
    return category_mappings


def remove_rows_from_restaurant_if_contains_categories(restaurants) -> List[dict]:
    """
    Returns restaurants with rows removed if they contain any of the categories in categories_to_remove
    """
    print("Number of restaurants before category filtering: ", len(restaurants))
    filtered_restaurants = []
    mappings = create_common_mappings()
    categories_to_remove = get_categories_to_remove()

    # get cats for each restaurant and check if any are in categories_to_remove
    for r in restaurants:
        cats = r.get('categories', [])
        cats = cats.split(', ') if isinstance(cats, str) else cats
        cats = [cat.strip().lower() for cat in cats]

        # check if any category in cats is in categories_to_remove if no, then we will continue logic abd eventualy append it
        if not any(cat in categories_to_remove for cat in cats):
            for c in cats:
                for key in mappings.keys():
                    if c in mappings[key]:
                        r[key] = True
            filtered_restaurants.append(r)
    print("Number of restaurants after category filtering: ", len(filtered_restaurants))
    print("Total number removed: ", len(restaurants) - len(filtered_restaurants))
    return filtered_restaurants


# add attributes as fied to restaurant dict
def add_attributes_to_restaurants(restaurants: List [dict], attribute_keys: List[str]):
    for r in restaurants:
        attrs = r.get('attributes', {})
        if attrs:
            for key in attribute_keys:
                value = attrs.get(key)
                if value is not None:
                    # convert string "True"/"False" to boolean
                    if isinstance(value, str):
                        if value.lower() == 'true' or 'true' in value.lower():
                            r[key] = True
                        elif value.lower() == 'false' or 'false' in value.lower():
                            r[key] = False
                        else:
                            r[key] = value
                    else:
                        r[key] = value
    return restaurants


def get_business_sentiment_scores(reviews: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average sentiment scores for each business based on its reviews.

    Args:
        reviews: List of review dicts
    """
    # agg mean compound sentiment scores at restaurant level
    business_sentiment_summary = reviews.groupby('business_id').agg({'vader_compound': 'mean', }).reset_index()
    # rename the column to be more descriptive
    business_sentiment_summary.columns = ['business_id', 'avg_sentiment']
    # merge sentiment summary back to restaurant data
    restaurants = restaurants.merge(business_sentiment_summary, on='business_id', how='left')
    return restaurants


def impute_values_for_cols(rdf):
    # update cols by using fillna with False for the bool columna
    bool_columns = ['has_coffee_or_tea', 'serves_sweets',
                    'serves_alcohol', 'DogsAllowed', 'has_european_food', 'has_american',
                    'has_asian_food', 'has_seafood', 'has_entertainment',
                    'has_hispanic_food', 'has_vegetarian', 'Open24Hours']

    for col in bool_columns:
        if col in rdf.columns:
            rdf[col] = rdf[col].fillna(False)
        else:
            print(col, " not in rdf columns")

    rdf['RestaurantsPriceRange2'] = rdf['RestaurantsPriceRange2'].fillna(2)

    return rdf


def get_local_pref_ratio(reviews, rdf, users):
    # use the reviews from review classification file to get local preference ratio
    # local preference ratio = num reviews from users in same city as restaurant / total num reviews for restaurant
    # create lookup dictionaries once
    business_city_map = rdf.set_index('business_id')['city'].to_dict()
    user_city_map = {u['user_id']: u.get('city') for u in users}

    # add city cols to reviews df
    reviews['business_city'] = reviews['business_id'].map(business_city_map)
    reviews['user_city'] = reviews['user_id'].map(user_city_map)

    # calc local preference ratio
    reviews['is_local'] = (reviews['business_city'] == reviews['user_city']).astype(int)

    local_pref_ratios = (
        reviews.groupby('business_id')['is_local']
        .agg(['sum', 'count'])
        .assign(local_preference_ratio=lambda x: x['sum'] / x['count'])
        ['local_preference_ratio'].to_dict())

    # add to restaurant df
    rdf['local_preference_ratio'] = rdf['business_id'].map(local_pref_ratios)

    return rdf


def get_users_df_with_agged_sentiment(reviews, users):
    """
    Create a users DataFrame with aggregated sentiment scores from their reviews.

    Args:
        reviews: DataFrame containing reviews with 'user_id' and 'vader_compound' columns
        users: List of user dicts to convert to DataFrame
    Returns:
        DataFrame with user_id and avg_sentiment columns
    """
    # agg mean compound sentiment scores at user level
    user_sentiment_summary = reviews.groupby('user_id').agg({'vader_compound': 'mean', }).reset_index()
    # Rename the column to be more descriptive
    user_sentiment_summary.columns = ['user_id', 'avg_sentiment']
    udf = pd.DataFrame(users)
    udf = udf.merge(user_sentiment_summary, on='user_id', how='left')
    return udf


def get_elite_users_features(reviews, udf, rdf):
    udf['years_elite'] = udf['elite'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x else 0)

    # Create elite user lookup (users who have been elite at least once)
    elite_user_ids = set(udf[udf['years_elite'] > 0]['user_id'])

    # add is_elite flag to reviews DataFrame
    reviews['is_elite'] = reviews['user_id'].isin(elite_user_ids)

    print("Retrieved elite reviews...")
    # calc elite user ratio using aggregation
    elite_user_ratios = (
        reviews.groupby('business_id')['is_elite']
        .agg(['sum', 'count'])
        .assign(elite_user_ratio=lambda x: x['sum'] / x['count'])
        ['elite_user_ratio'].to_dict())
    print("Calculated elite user ratios...")

    # Add to restaurant dataframe
    rdf['elite_user_ratio'] = rdf['business_id'].map(elite_user_ratios).fillna(0)

    return rdf, udf, reviews


# def get_user_features(reviews, users: List[dict]) -> pd.DataFrame:
#     """
#     Convert list of user dicts to DataFrame.
#
#     Args:
#         users: List of user dicts
#     Returns:
#         DataFrame with user features
#     """
#     udf = get_users_df_with_agged_sentiment(reviews, users)
#     return udf


def get_user_cat_counts(rdf, reviews):
    # Create business_id -> categories mapping once
    business_categories = rdf.set_index('business_id')['categories'].to_dict()

    # category counts per user
    user_category_counts = defaultdict(Counter)

    # process categories for all reviews at once
    for user_id, business_id in zip(reviews['user_id'], reviews['business_id']):
        cats = business_categories.get(business_id)

        if cats:
            cats = cats.split(', ') if isinstance(cats, str) else cats

            for cat in cats:
                cat_lower = cat.strip().lower()
                # avoid the obvious/top categories

                if cat_lower not in ('food', 'restaurants'):
                    user_category_counts[user_id][cat_lower] += 1
    return user_category_counts


def get_ratios(rdf, reviews, udf):
    user_category_counts = get_user_cat_counts(rdf, reviews)
    user_category_ratios = {}
    category_mappings = create_common_mappings()
    for user_id, category_counter in user_category_counts.items():
        if not category_counter:
            continue

        total_reviews = sum(category_counter.values())
        user_ratios = {}

        # for each mapped category group (e.g., 'has_american', 'has_asian_food')
        for mapping_key, category_list in category_mappings.items():
            # count reviews in this category group
            count = sum(category_counter[cat] for cat in category_list if cat in category_counter)
            ratio = count / total_reviews if total_reviews > 0 else 0
            user_ratios[mapping_key + "_ratio"] = ratio

        user_category_ratios[user_id] = user_ratios

    # add to user category ratios to user dataframe
    for mapping_key in category_mappings.keys():
        updated_key = mapping_key + "_ratio"
        udf[updated_key] = udf['user_id'].map(lambda uid: user_category_ratios.get(uid, {}).get(updated_key, 0))

    udf = get_user_top_category(user_category_counts, udf)

    return udf

def get_user_top_category(user_category_counts, udf):
    # get each user's top category and ratio
    user_top_categories = {}

    for user_id, category_counter in user_category_counts.items():
        if not category_counter:
            continue

        # get most common category
        top_category, count = category_counter.most_common(1)[0]

        # calculate ratio
        total_reviews = sum(category_counter.values())
        ratio = count / total_reviews if total_reviews > 0 else 0

        user_top_categories[user_id] = {
            'top_category': top_category,
            'count': count,
            'total_reviews': total_reviews,
            'ratio': ratio}
    # add to user df
    udf['top_category'] = udf['user_id'].map(lambda uid: user_top_categories.get(uid, {}).get('top_category'))
    udf['top_category_ratio'] = udf['user_id'].map(lambda uid: user_top_categories.get(uid, {}).get('ratio', 0))
    return udf


def get_user_rating_variance(reviews, udf):
    # user rating variance
    user_rating_stats = reviews.groupby('user_id')['stars'].agg(['std']).reset_index()
    user_rating_stats.columns = ['user_id', 'rating_variance']
    udf = udf.merge(user_rating_stats, on='user_id', how='left')
    return udf


def get_user_locality_pref_ratio(reviews, udf):
    # calculate local vs tourist ratio for users using reviews
    user_local_ratios = (
        reviews.groupby('user_id')['is_local']
        .agg(['sum', 'count'])
        .assign(local_review_ratio=lambda x: x['sum'] / x['count'])
        ['local_review_ratio']
        .to_dict())

    udf['local_review_ratio'] = udf['user_id'].map(user_local_ratios).fillna(0)
    return udf


def get_date_features(reviews, udf):
    # review recency and frequency
    reviews['date'] = pd.to_datetime(reviews['date'])
    user_review_activity = reviews.groupby('user_id')['date'].agg(
        [('first_review', 'min'), ('last_review', 'max')]).reset_index()

    # create user activity features
    user_review_activity['days_since_last_review'] = (reviews.date.max() - user_review_activity['last_review']).dt.days
    user_review_activity['days_active'] = (
                user_review_activity['last_review'] - user_review_activity['first_review']).dt.days
    user_review_activity['review_frequency'] = udf.set_index('user_id')['review_count'] / user_review_activity[
        'days_active'].replace(0, 1)

    udf = udf.merge(user_review_activity[['user_id', 'days_since_last_review', 'review_frequency']], on='user_id',
                    how='left')
    return udf, reviews


def get_price_pref(rdf, reviews, udf):
    # price range preference
    user_price_prefs = reviews.merge(
        rdf[['business_id', 'RestaurantsPriceRange2']],
        on='business_id',
        how='left')

    # convert to numeric, coercing errors to NaN
    user_price_prefs['RestaurantsPriceRange2'] = pd.to_numeric(user_price_prefs['RestaurantsPriceRange2'],
                                                               errors='coerce').fillna(2)

    # calculate mean, ignoring NaN values
    user_avg_price = user_price_prefs.groupby('user_id')['RestaurantsPriceRange2'].mean().to_dict()
    udf['avg_price_preference'] = udf['user_id'].map(user_avg_price)
    return udf

def get_geo_diversity(reviews, udf):
    # geographic diversity
    user_city_diversity = reviews.groupby('user_id')['business_city'].nunique().to_dict()
    udf['num_cities_reviewed'] = udf['user_id'].map(user_city_diversity).fillna(1)
    return udf

def get_business_features():
    pass