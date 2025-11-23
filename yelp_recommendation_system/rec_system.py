import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from .model import HybridRecommender
from .preprocessing import (
    load_data,
    filter_review_data,
    filter_business_data,
    get_available_restaurant_columns,
    preprocess_business_features,
    preprocess_user_features)
from .evaluation import evaluate_model, print_evaluation_results
from .utils import save_model, load_model, get_user_recommendations, get_batch_predictions, create_train_test_sets


class YelpRecommenderSystem:
    """
    High-level interface for the Yelp Rec Sys.

    This class provides a simple API for:
    - Loading and preprocessing data
    - Training hybrid recommendation models
    - Generating personalized recommendations
    - Evaluating model performance
    - Saving and loading models

    """

    def __init__(self):
        self.model = None
        self.trainset = None
        self.testset = None
        self.rdf = None
        self.business_df = None
        self.user_df = None
        self.business_features = None
        self.user_features = None
        self.filtered_reviews = None
        self.filtered_business_df = None

    def load_data(
        self,
        user_path: str,
        restaurant_path: str,
        reviews_path: str):
        """
        Load user, restaurant, and review data from CSV files.

        Args:
            user_path: Path to user features CSV file
            restaurant_path: Path to restaurant features CSV file
            reviews_path: Path to reviews CSV file
        """
        self.rdf, self.business_df, self.user_df = load_data(user_path, restaurant_path, reviews_path)

    def train(
        self,
        n_factors: int = 100,
        n_epochs: int = 10,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        alpha: float = 0.001,
        beta: float = 0.001,
        min_user_reviews: int = 20,
        min_business_reviews: int = 50,
        test_size: float = 0.2,
        random_state: int = 42) -> Dict[str, Any]:
        """
        Train the hybrid recommendation model.

        This method performs the complete training pipeline:
        1. Filter reviews and businesses
        2. Preprocess features
        3. Create train/test splits
        4. Train the hybrid model
        5. Evaluate on test set

        Args:
            n_factors: Number of latent factors for matrix factorization
            n_epochs: Number of training iterations
            learning_rate: Learning rate for gradient descent
            regularization: L2 regularization coefficient
            alpha: Weight for user features component
            beta: Weight for business features component
            min_user_reviews: Minimum reviews per user to include
            min_business_reviews: Minimum reviews per restaurant to include
            test_size: Proportion of data for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility

        Returns:
            dict: Training history with epoch-wise metrics
        """
        if self.rdf is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # filter data
        print("\n" + "="*60)
        print("STEP 1: Filtering Data")
        print("="*60)
        self.filtered_reviews = filter_review_data(self.rdf, min_user_reviews=min_user_reviews, min_rest_reviews=min_business_reviews)

        # preprocess business features
        print("\n" + "="*60)
        print("STEP 2: Preprocessing Business Features")
        print("="*60)
        self.filtered_business_df = filter_business_data(self.business_df, self.filtered_reviews)

        available_numeric, available_categorical, available_info = \
            get_available_restaurant_columns(self.filtered_business_df)

        self.business_features = preprocess_business_features(self.filtered_business_df, available_numeric, available_categorical)

        # preprocess user features
        print("\n" + "="*60)
        print("STEP 3: Preprocessing User Features")
        print("="*60)
        self.user_features = preprocess_user_features(self.filtered_reviews, self.user_df)

        # create train/test splits
        print("\n" + "="*60)
        print("STEP 4: Creating Train/Test Splits")
        print("="*60)
        self.trainset, self.testset = create_train_test_sets(self.filtered_reviews, test_size=test_size, random_state=random_state)

        # train the model
        print("\n" + "="*60)
        print("STEP 5: Training Hybrid Model")
        print("="*60)
        print(f"Configuration:")
        print(f"  Latent factors: {n_factors}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Regularization: {regularization}")
        print(f"  Alpha (user features): {alpha}")
        print(f"  Beta (business features): {beta}")
        print()

        self.model = HybridRecommender(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=learning_rate,
            reg=regularization,
            user_features=self.user_features,
            loc_features=self.business_features,
            alpha=alpha,
            beta=beta)

        self.model.fit_and_track(self.trainset)

        # evaluate on test set
        print("\n" + "="*60)
        print("STEP 6: Evaluating Model")
        print("="*60)
        predictions = self.model.test(self.testset)
        metrics = evaluate_model(predictions)
        print_evaluation_results(metrics)

        # store metrics in history
        self.model.history['test_metrics'] = metrics

        return self.model.history

    def get_recommendations(self, user_id: str, n_recommendations: int = 10, min_rating: float = 3.5) -> pd.DataFrame:
        """
        Get top N restaurant recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            min_rating: Minimum predicted rating threshold

        Returns:
            DataFrame with recommended restaurants and predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return get_user_recommendations(
            self.model,
            user_id,
            self.business_features,
            self.filtered_reviews,
            self.filtered_business_df,
            n=n_recommendations,
            min_rating=min_rating)

    def predict_ratings(self, user_business_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Predict ratings for specific user-restaurant pairs.

        Args:
            user_business_pairs: List of (user_id, business_id) tuples

        Returns:
            DataFrame with predicted ratings for each pair

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return get_batch_predictions(self.model, user_business_pairs)


    def evaluate(self, testset=None) -> Dict[str, float]:
        """
        Evaluate model performance on a test set.
        Args:
            testset: Optional custom test set. If None, uses the test set from training.
        Returns:
            dict: Dictionary of evaluation metrics
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if testset is None:
            if self.testset is None:
                raise ValueError("No test set available. Provide testset or call train() first.")
            testset = self.testset

        predictions = self.model.test(testset)
        metrics = evaluate_model(predictions)
        print_evaluation_results(metrics)

        return metrics


    def save_model(self, filepath: str):
        """
        Save the trained model and all supporting data to disk.

        This saves:
        - The trained model
        - Business features
        - User features
        - Filtered reviews
        - Filtered business dataframe
        - Trainset and testset

        Args:
            filepath: Path where the model should be saved
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # create a bundle with all necessary data
        model_bundle = {
            'model': self.model,
            'business_features': self.business_features,
            'user_features': self.user_features,
            'filtered_reviews': self.filtered_reviews,
            'filtered_business_df': self.filtered_business_df,
            'trainset': self.trainset,
            'testset': self.testset}

        save_model(model_bundle, filepath)

    def load_model(self, filepath: str):
        """
        Load a previously saved model and all supporting data.

        This restores:
        - The trained model
        - Business features
        - User features
        - Filtered reviews
        - Filtered business dataframe
        - Trainset and testset

        Args:
            filepath: Path to the saved model file
        Returns:
            self: The YelpRecommenderSystem instance with loaded model
        """
        loaded_data = load_model(filepath)

        # check if it's a bundle
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            # bundle with all data
            self.model = loaded_data['model']
            self.business_features = loaded_data.get('business_features')
            self.user_features = loaded_data.get('user_features')
            self.filtered_reviews = loaded_data.get('filtered_reviews')
            self.filtered_business_df = loaded_data.get('filtered_business_df')
            self.trainset = loaded_data.get('trainset')
            self.testset = loaded_data.get('testset')
            print("Loaded model bundle with all supporting data")
        else:
            # just the model
            self.model = loaded_data
            print("You will need to load data separately to generate recommendations.")

        return self


    def save_bundle(self, filepath: str):
        """
        Save model as a bundle (alias for save_model).

        This is an explicit alias for save_model() to make it clear that
        a complete bundle with all components is being saved.

        The bundle includes:
        - Trained model
        - Business features
        - User features
        - Filtered reviews
        - Filtered business dataframe
        - Trainset and testset

        Args:
            filepath: Path where the bundle should be saved
        """
        self.save_model(filepath)
        print(f" Bundle saved with all components to: {filepath}")

    def load_bundle(self, filepath: str):
        """
        Load model from a bundle (alias for load_model).

        This is an explicit alias for load_model() to make it clear that
        a complete bundle is being loaded.

        Args:
            filepath: Path to the bundle file

        Returns:
            self: The YelpRecommenderSystem instance with loaded model

        """
        return self.load_model(filepath)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        Returns:
            dict: Dictionary containing model configuration and statistics
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return {
            'n_factors': self.model.n_factors,
            'n_epochs': self.model.n_epochs,
            'learning_rate': self.model.lr,
            'regularization': self.model.reg,
            'alpha': self.model.alpha,
            'beta': self.model.beta,
            'n_users': self.trainset.n_users if self.trainset else None,
            'n_items': self.trainset.n_items if self.trainset else None,
            'n_ratings': self.trainset.n_ratings if self.trainset else None,
            'has_user_features': self.model.has_user_features,
            'has_business_features': self.model.has_loc_features}


