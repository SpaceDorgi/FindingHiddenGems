# business_id specific hidden gem score
# Using PySpark for distributed processing and scikit-learn for ML-based classification
# gives each business a hidden_gem_score based on rules
# Version: 2.0
# pip install pyspark scikit-learn pandas numpy joblib
#!pip install pyspark

import logging
import json
import joblib
import sqlite3
import os
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from pyspark.sql.functions import expr

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, udf, count, avg, min, max, when, collect_list,
    row_number, lag, datediff, dayofweek, month, year, explode,
    coalesce, lit, abs as spark_abs, approx_count_distinct, stddev, greatest, first
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType,
    ArrayType, DoubleType
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('local_tourist_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)


class LocalTouristAnalyzer:
    """
    Production-ready analyzer for classifying Yelp reviewers as locals or tourists.
    Uses PySpark for distributed processing and ML models for classification.
    """

    def __init__(self, spark: SparkSession = None, config: Dict = None):
        """
        Initialize the analyzer with optional Spark session and configuration.

        Args:
            spark: SparkSession instance. If None, creates a new session.
            config: Configuration dictionary with thresholds and parameters.
        """
        self.spark = spark
        assert self.spark is not None, "A SparkSession must be provided"
        self.logger = logger
        self.config = config or self._default_config()

        # Data storage
        self.users_df = None
        self.reviews_df = None
        self.business_df = None
        self.user_features_df = None
        self.classifications_df = None
        self.business_metrics_df = None # Added to store business metrics at business level


        # Models
        self.ml_model = None
        self.model_path = None
        self.scaler = None

        self.logger.info("LocalTouristAnalyzer initialized with config: %s", json.dumps(self.config, indent=2))


    @staticmethod
    def _default_config() -> Dict:
        """Return default configuration parameters."""
        return {
            'local_indicators': {
                'review_frequency': {'high': 50, 'medium': 20},  # reviews per year
                'geographic_scope': {'local_radius': 25, 'regional_radius': 100},  # miles
                'temporal_patterns': {'consistent_activity': 12, 'seasonal_variation': 0.3},
                'social_indicators': {'friends_threshold': 10, 'compliment_ratio': 0.1}
            },
            'tourist_indicators': {
                'review_clusters': {'short_timespan': 7},  # days
                'geographic_spread': 100,  # miles
                'business_types': ['Hotels', 'Tours', 'Attractions', 'Fine Dining'],
                'seasonal_patterns': {'vacation_months': [6, 7, 8, 11, 12], 'weekend_bias': 0.7}
            },
            'model_type': 'random_forest',  # 'random_forest', 'gradient_boosting', 'logistic_regression'
            'test_size': 0.2,
            'random_state': 42,
            'batch_size': 5000  # for batch processing
        }

    def load_data(self, sqlite_db_path: str):
        """
        Load data from SQLite filtered views filtered_business, filtered_reviews, filtered_users
        and convert them into Spark DataFrames.
        """
        try:
            self.logger.info(f"Loading data from SQLite views at {sqlite_db_path}...")

            conn = sqlite3.connect(sqlite_db_path)

            # Load filtered data into pandas DataFrames
            business_pdf = pd.read_sql_query("SELECT * FROM business", conn)
            reviews_pdf = pd.read_sql_query("SELECT * FROM reviews", conn)
            users_pdf = pd.read_sql_query("SELECT * FROM users", conn)

            conn.close()

            # Convert Pandas DataFrames to Spark DataFrames
            self.business_df = self.spark.createDataFrame(business_pdf)
            self.reviews_df = self.spark.createDataFrame(reviews_pdf)
            self.users_df = self.spark.createDataFrame(users_pdf)

            user_count = self.users_df.count()
            review_count = self.reviews_df.count()
            business_count = self.business_df.count()

            self.logger.info(f"Loaded {user_count} users, {review_count} reviews, {business_count} businesses from SQLite")

            return {
                'users': user_count,
                'reviews': review_count,
                'businesses': business_count
            }

        except Exception as e:
            self.logger.error(f"Error loading data from SQLite: {e}")
            raise


    def engineer_features(self) -> DataFrame:
        """
        Engineer features for local/tourist classification from raw data.

        Features include:
        - Review frequency (reviews per year)
        - Geographic scope (radius of reviews, unique cities)
        - Temporal patterns (consistency, seasonal variation, weekend bias)
        - Social patterns (friend count, elite status, compliments)
        - Business type preferences

        Returns:
            DataFrame with engineered features per user
        """
        self.logger.info("Engineering features...")

        try:
            # Calculate account age and review frequency (User-level)
            review_stats = self.reviews_df.groupBy('user_id').agg(
                count('*').alias('num_reviews'),
                avg('stars').alias('avg_stars'),
                max('date').alias('latest_review_date'),
                min('date').alias('first_review_date')
            )

            # Geographic features - join with business data to get locations (Review-level)
            reviews_with_geo = self.reviews_df.select(
                'user_id', 'business_id', 'date', 'stars'
            ).join(
                self.business_df.select('business_id', 'name','lat', 'lon', 'city'),
                'business_id',
                'left'
            )


            # Calculate user's most frequent review location (User-level based on Review-level data)
            user_centers = reviews_with_geo.groupBy('user_id', 'city').agg(
                count('*').alias('reviews_in_city')
            ).withColumn(
                'row_num',
                row_number().over(Window.partitionBy('user_id').orderBy(col('reviews_in_city').desc()))
            ).filter(col('row_num') == 1).select('user_id', col('city').alias('home_city'))

            # Join reviews with home city information
            reviews_with_home = reviews_with_geo.join(user_centers, 'user_id', 'left')


            # Calculate geographic features at the user level by aggregating from reviews_with_home
            user_geo_features = reviews_with_home.groupBy('user_id').agg(
                 approx_count_distinct('city').alias('unique_cities'),
                 count(when(col('lat').isNotNull(), 1).otherwise(None)).alias('reviews_with_location'),
                 min('lat').alias('min_lat'),
                 max('lat').alias('max_lat'),
                 min('lon').alias('min_lon'),
                 max('lon').alias('max_lon'),
                 avg(when(col('city') == col('home_city'), 1).otherwise(0)).alias('home_city_review_ratio')
            )


            # Calculate business-level metrics (Business-level)#this is for the Vizu module
            business_metrics = reviews_with_home.groupBy('business_id').agg(
                count('*').alias('total_reviews'),
                avg('stars').alias('avg_rating'),
                count(when(col('city') == col('home_city'), 1).otherwise(None)).alias('local_reviews'),
                count(when(col('city') != col('home_city'), 1).otherwise(None)).alias('tourist_reviews'),
                first('lat').alias('lat'),
                first('lon').alias('lon'),
                first('name').alias('name')
            ).withColumn(
                'local_ratio',
                col('local_reviews') / col('total_reviews')
            ).withColumn(
                'sentiment_score',
                when(col('avg_rating') >= 3.5, (col('avg_rating') - 2.5) / 2.5).otherwise(0)
            ).withColumn(
                'volume_score',
                expr('exp(-total_reviews / 200)')  # Exponential decay for review volume
            ).withColumn(
                'local_preference_score',
                pow(col('local_ratio'), 1.5)
            ).withColumn(
                'hidden_gem_score',
                ((col('sentiment_score') * 0.30) +  # High local sentiment
                 (col('volume_score') * 0.25) +     # Low mainstream attention
                 (col('local_preference_score') * 0.20) +  # Local preference strength
                 when(col('avg_rating') >= 3.5, 1.0).otherwise(pow(col('avg_rating') / 3.5, 2)) * 0.25  # Quality threshold
                ).cast('double')
            )
            self.business_metrics_df = business_metrics # Store business metrics


            # Temporal features (User-level)
            temporal_features = self.reviews_df.groupBy('user_id').agg(
                avg(dayofweek('date')).alias('avg_review_dayofweek'),  # 1=Sunday, 7=Saturday
                stddev(month('date')).alias('seasonal_variation')
            )

            # Social features (User-level)
            user_social = self.users_df.select(
                'user_id',
                col('friends').alias('num_friends'),
                col('elite').alias('elite_status'),
                col('compliment_hot').alias('compliment_hot'),
                col('compliment_cool').alias('compliment_cool'),
                col('compliment_funny').alias('compliment_funny'),
                col('fans').alias('num_fans')
            )

            # Combine all user-level features
            user_features = review_stats \
                .join(user_geo_features, 'user_id', 'left') \
                .join(temporal_features, 'user_id', 'left') \
                .join(user_social, 'user_id', 'left') \
                .fillna(0)

            # Add derived features (User-level)
            user_features = user_features.withColumn(
                'reviews_per_year',
                col('num_reviews') / (
                    (datediff(col('latest_review_date'), col('first_review_date')) / 365.25) + 0.5
                )
            ).withColumn(
                'geographic_radius',
                when(
                    col('min_lat').isNotNull() &
                    col('min_lon').isNotNull() &
                    col('max_lat').isNotNull() &
                    col('max_lon').isNotNull(),
                    self._calculate_haversine_udf()(
                        col('min_lat'), col('min_lon'),
                        col('max_lat'), col('max_lon')
                    )
                ).otherwise(0.0)
            ).withColumn(
                'total_compliments',
                col('compliment_hot') + col('compliment_cool') + col('compliment_funny')
            ).withColumn(
                'compliment_ratio',
                col('total_compliments') / (col('num_reviews') + 1)
            ).withColumn(
                'elite_indicator',
                when(col('elite_status').isNotNull() & (col('elite_status') > 0), 1).otherwise(0)
            )

            self.user_features_df = user_features
            feature_count = user_features.count()


            self.logger.info(f"Engineered features for {feature_count} users")


            return user_features # Return user-level features

        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            raise

    @staticmethod
    def _calculate_haversine_udf():
        """Create UDF for haversine distance calculation."""
        def haversine(lat1, lon1, lat2, lon2):
            if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                return 0.0
            R = 3959  # Earth radius in miles
            dLat = radians(lat2 - lat1)
            dLon = radians(lon2 - lon1)
            a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return float(R * c)

        return udf(haversine, DoubleType())

    def save_results(self, output_path: str, format: str = 'parquet'):
        """
        Save classification results and business metrics to files.

        Args:
            output_path: Base path to save results
            format: 'parquet', 'csv', or 'json'
        """
        try:
            if self.classifications_df is None:
                raise ValueError("No classifications to save.")

            # Create subdirectories for different types of results
            users_path = f"{output_path}/users"
            business_path = f"{output_path}/businesses"
            os.makedirs(users_path, exist_ok=True)
            os.makedirs(business_path, exist_ok=True)

            # Save user classifications
            if format == 'parquet':
                self.classifications_df.write.parquet(users_path, mode='overwrite')
                if self.business_metrics_df is not None:
                    self.business_metrics_df.write.parquet(business_path, mode='overwrite')
            elif format == 'csv':
                self.classifications_df.coalesce(1).write.csv(users_path, mode='overwrite', header=True)
                if self.business_metrics_df is not None:
                    self.business_metrics_df.coalesce(1).write.csv(business_path, mode='overwrite', header=True)
            elif format == 'json':
                if self.business_metrics_df is not None:
                    business_json = self.business_metrics_df.toJSON().collect()
                    json_path = f"{business_path}/business_metrics.json"
                    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                        with open(f"{business_path}/business_metrics.json", 'w') as f:
                            json.dump([json.loads(row) for row in business_json], f, indent=2)
                    else:
                        with open(json_path, 'r') as f:
                            existing_data = json.load(f)
                        combined_data = existing_data + [json.loads(row) for row in business_json]
                        with open(json_path, 'w') as f:
                            json.dump(combined_data, f, indent=2)
            
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"User classifications saved to {users_path} in {format} format")
            if self.business_metrics_df is not None:
                self.logger.info(f"Business metrics saved to {business_path} in {format} format")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise


# Example usage and entry point
if __name__ == "__main__":
    print("LocalTouristAnalyzer Start!")

    # Mount Google Drive
    try:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        raise

    try:
        # Initialize Spark session with optimized settings for Colab
        spark = SparkSession.builder \
            .appName("LocalTouristAnalyzer") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()

        # Initialize analyzer
        analyzer = LocalTouristAnalyzer(spark=spark)

        # Load business_ids from text file
        with open("/content/drive/My Drive/Colab/datasql/data/business_ids.txt", "r") as f:
            business_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(business_ids)} business_ids from business_ids.txt")

        # Load data from SQLite database in Google Drive
        print(f"DB path used: {sqlite_db_path}")
        data_stats = analyzer.load_data(sqlite_db_path=sqlite_db_path)
        print("Data loaded:", data_stats)

        # Filter business_df and reviews_df to only those business_ids
        analyzer.business_df = analyzer.business_df.filter(col("business_id").isin(list(business_ids)))
        analyzer.reviews_df = analyzer.reviews_df.filter(col("business_id").isin(list(business_ids)))

        # Engineer features
        print("Engineering features...")
        features = analyzer.engineer_features()
        print("Features engineered successfully")

        # Save results to Google Drive
        output_path = "/content/drive/My Drive/Colab/datasql/output/classifications"
        print(f"Saving results to {output_path}")
        analyzer.save_results(output_path, format="json")
        print(f"Results saved successfully to:")
        print(f"- User classifications: {output_path}/users")
        print(f"- Business metrics: {output_path}/businesses")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()
            print("Spark session stopped")

    print("LocalTouristAnalyzer End!")