import numpy as np
from tqdm import tqdm
from surprise import AlgoBase, PredictionImpossible
from .timer import timeit
import warnings
warnings.filterwarnings('ignore')


class HybridRecommender(AlgoBase):
    """
    Hybrid recommendation system combining:
    - Matrix Factorization (collaborative filtering via SVD)
    - User feature-based predictions
    - Business feature-based predictions

    The model learns latent factors for users and items while also learning
    how to project user and business features into the latent space.

    Attributes:
        n_factors (int): Number of latent factors for matrix factorization
        n_epochs (int): Number of training iterations
        lr (float): Learning rate for latent factors and biases
        lr_W (float): Learning rate for feature projection matrices
        reg (float): Regularization coefficient for MF components
        reg_W (float): Regularization coefficient for feature matrices
        alpha (float): Weight for user features component
        beta (float): Weight for business features component
    """

    def __init__(
        self,
        n_factors=100,
        n_epochs=10,
        lr=0.005,
        reg=0.02,
        user_features=None,
        loc_features=None,
        alpha=1e-3,
        beta=1e-3,
        lr_W=None,
        user_feat_matrix=None,
        loc_feat_matrix=None,):
        """
        Initialize the hybrid recommender.

        Args:
            n_factors: Number of latent factors for matrix factorization
            n_epochs: Number of training iterations
            lr: Learning rate for MF components (latent factors and biases)
            reg: L2 regularization coefficient for MF components
            user_features: DataFrame with user features (includes user_id column)
            loc_features: DataFrame with business/location features (includes business_id column)
            alpha: Weight for user features component (0 to disable)
            beta: Weight for business features component (0 to disable)
            lr_W: Learning rate for feature projection matrices (defaults to lr/10)
            user_feat_matrix: Pre-computed user feature matrix (for faster experimentation)
            loc_feat_matrix: Pre-computed business feature matrix (for faster experimentation)
        """
        AlgoBase.__init__(self)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_W = lr_W if lr_W is not None else lr / 10.0
        self.reg = reg
        self.reg_W = 0.01
        self.alpha = alpha
        self.beta = beta

        self.user_features_df = user_features
        self.loc_features_df = loc_features
        self.cached_user_feat_matrix = user_feat_matrix
        self.cached_loc_feat_matrix = loc_feat_matrix

    @timeit("fit recommender")
    def fit_and_track(self, trainset):
        """
        This method fits the model and tracks training metrics (RMSE, MAE) over epochs.

        Args:
            trainset: Surprise trainset object containing user-item ratings

        Returns:
            self: The fitted model instance
        """
        # initialize the trainset (required by Surprise)
        AlgoBase.fit(self, trainset)

        # initialize latent factor matrices with small random values
        self.pu = np.random.normal(0, 0.1, (trainset.n_users, self.n_factors))
        self.qi = np.random.normal(0, 0.1, (trainset.n_items, self.n_factors))

        # initialize bias terms
        self.bu = np.zeros(trainset.n_users)
        self.bi = np.zeros(trainset.n_items)
        self.global_mean = trainset.global_mean

        # prepare and normalize feature matrices
        self._prepare_features(trainset)

        # initialize training history for tracking performance
        self.history = {"epoch": [], "train_rmse": [], "train_mae": []}

        # gradient clipping values to prevent explosion
        clip_val = 5.0
        clip_val_W = 5.0

        # training loop
        for epoch in tqdm(range(self.n_epochs), desc="Training", leave=False):
            sq_errors = []
            abs_errors = []

            # iterate through all ratings in random order
            for u, i, r in trainset.all_ratings():
                # get current prediction
                pred = self._get_prediction(u, i)

                # safety check for numerical stability
                if not np.isfinite(pred):
                    print(f"Non-finite pred at epoch {epoch}, u={u}, i={i}")
                    print("pu[u]:", self.pu[u])
                    print("qi[i]:", self.qi[i])
                    raise ValueError("Non-finite prediction detected")

                # compute error
                err = r - pred
                sq_errors.append(err ** 2)
                abs_errors.append(abs(err))

                # update latent factors using stochastic gradient descent (SGD
                temp_pu = self.pu[u].copy()
                self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                self.qi[i] += self.lr * (err * temp_pu - self.reg * self.qi[i])

                # clip params to prevent explosion
                self.pu[u] = np.clip(self.pu[u], -clip_val, clip_val)
                self.qi[i] = np.clip(self.qi[i], -clip_val, clip_val)

                # update bias terms
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # update user feature projection matrix
                if self.has_user_features and self.alpha > 0.0:
                    grad_Wu = (np.outer(self.user_feat_matrix[u], err * self.qi[i] * self.alpha) - self.reg_W * self.Wu)
                    self.Wu += self.lr_W * grad_Wu
                    self.Wu = np.clip(self.Wu, -clip_val_W, clip_val_W)

                # update business feature projection matrix
                if self.has_loc_features and self.beta > 0.0:
                    grad_Wi = (np.outer(self.loc_feat_matrix[i], err * self.pu[u] * self.beta) - self.reg_W * self.Wi)
                    self.Wi += self.lr_W * grad_Wi
                    self.Wi = np.clip(self.Wi, -clip_val_W, clip_val_W)

            # calculate end-of-epoch metrics
            if len(sq_errors) == 0:
                epoch_rmse = np.nan
                epoch_mae = np.nan
            else:
                epoch_rmse = np.sqrt(np.mean(sq_errors))
                epoch_mae = np.mean(abs_errors)

            # log metrics
            self.history["epoch"].append(epoch + 1)
            self.history["train_rmse"].append(epoch_rmse)
            self.history["train_mae"].append(epoch_mae)

        return self


    @timeit("prepare user and restaurant features")
    def _prepare_features(self, trainset):
        """
        Prepare and normalize user and business feature matrices.

        This method converts feature DataFrames into numpy arrays aligned with
        the trainset's internal user/item IDs.

        Args:
            trainset: Surprise trainset object
        """
        # prep user features
        if self.cached_user_feat_matrix is not None:
            # Use pre-computed feature matrix if provided
            self.user_feat_matrix = np.nan_to_num(self.cached_user_feat_matrix, nan=0.0)
            self.has_user_features = True
            n_user_feats = self.user_feat_matrix.shape[1]
            self.Wu = np.random.normal(0, 0.1, (n_user_feats, self.n_factors))

        elif self.user_features_df is not None:
            # exract feature columns (exclude user_id)
            user_feat_cols = [col for col in self.user_features_df.columns if col != 'user_id']
            self.user_feat_matrix = np.zeros((trainset.n_users, len(user_feat_cols)))

            # create dict for fast lookup
            user_dict = self.user_features_df.set_index('user_id')[user_feat_cols].to_dict('index')

            # map features to internal user id
            for inner_uid in range(trainset.n_users):
                raw_uid = trainset.to_raw_uid(inner_uid)
                if raw_uid in user_dict:
                    self.user_feat_matrix[inner_uid] = list(user_dict[raw_uid].values())

            # handle missing values and initialize projection matrix
            self.user_feat_matrix = np.nan_to_num(self.user_feat_matrix, nan=0.0)
            self.Wu = np.random.normal(0, 0.1, (len(user_feat_cols), self.n_factors))
            self.has_user_features = True

        else:
            # no user features provided
            self.has_user_features = False
            self.user_feat_matrix = None
            self.Wu = None

        # prep business/location features
        if self.cached_loc_feat_matrix is not None:
            # use pre-computed feature matrix if provided (only saves like 10 seconds but better than nothing)
            self.loc_feat_matrix = np.nan_to_num(self.cached_loc_feat_matrix, nan=0.0)
            self.has_loc_features = True
            n_loc_feats = self.loc_feat_matrix.shape[1]
            self.Wi = np.random.normal(0, 0.1, (n_loc_feats, self.n_factors))

        elif self.loc_features_df is not None:
            # extract feature columns (exclude business_id)
            loc_feat_cols = [col for col in self.loc_features_df.columns if col != 'business_id']
            self.loc_feat_matrix = np.zeros((trainset.n_items, len(loc_feat_cols)))

            # create dict for fast lookup
            loc_dict = self.loc_features_df.set_index('business_id')[loc_feat_cols].to_dict('index')

            # map features to internal loc or "item" ids
            for inner_iid in range(trainset.n_items):
                raw_iid = trainset.to_raw_iid(inner_iid)
                if raw_iid in loc_dict:
                    self.loc_feat_matrix[inner_iid] = list(loc_dict[raw_iid].values())

            # handle missing vals and initialize projection matrix
            self.loc_feat_matrix = np.nan_to_num(self.loc_feat_matrix, nan=0.0)
            self.Wi = np.random.normal(0, 0.1, (len(loc_feat_cols), self.n_factors))
            self.has_loc_features = True

        else:
            # no business features provided
            self.has_loc_features = False
            self.loc_feat_matrix = None
            self.Wi = None

    def _get_prediction(self, u, i):
        """
        Calculate prediction for user u and item i using the hybrid model.
        The prediction combines:
        1. Global mean + user bias + item bias
        2. Collaborative filtering (user latent and business latent: dot(pu, qi)
        3. User features: alpha * dot(user_features @ Wu, qi)
        4. Restaurant features: beta * dot(pu, item_features @ Wi)
        Args:
            u: Internal user ID
            i: Internal item ID
        Returns:
            float: Predicted rating
        """
        # start with baseline pred
        pred = self.global_mean + self.bu[u] + self.bi[i]

        # add collaborative filtering component
        pred += np.dot(self.pu[u], self.qi[i])

        # add user feature component if available
        if self.has_user_features and self.alpha > 0.0:
            user_latent_from_feats = self.user_feat_matrix[u] @ self.Wu
            pred += self.alpha * np.dot(user_latent_from_feats, self.qi[i])

        # add loc feature component if available
        if self.has_loc_features and self.beta > 0.0:
            loc_latent_from_feats = self.loc_feat_matrix[i] @ self.Wi
            pred += self.beta * np.dot(self.pu[u], loc_latent_from_feats)

        return pred

    def estimate(self, u, i):
        """
        Estimate rating for user u and item i (required by Surprise).
        Args:
            u: Internal user ID
            i: Internal item ID (business id)
        Returns:
            float: Predicted rating
        """
        if not self.trainset.knows_user(u) or not self.trainset.knows_item(i):
            raise PredictionImpossible("User or item unknown")

        return self._get_prediction(u, i)


    def predict_for_new_user(self, user_feat_vector, raw_item_id):
        # map raw business id -> internal id
        try:
            i = self.trainset.to_inner_iid(raw_item_id)
        except ValueError:
            raise ValueError("Unknown item/business")

        # build latent user vector from features
        user_feat_vector = np.nan_to_num(user_feat_vector, nan=0.0)
        user_latent = user_feat_vector @ self.Wu

        # get known restaurant parameters
        qi = self.qi[i]
        bi = self.bi[i]
        loc_latent = None
        if self.has_loc_features:
            loc_latent = self.loc_feat_matrix[i] @ self.Wi

        # make prediction
        pred = self.global_mean + bi  # no user bias yet
        pred += np.dot(user_latent, qi)  # main CF-ish signal from features

        if self.has_loc_features:
            # reuse user_latent in place of pu[u]
            pred += self.beta * np.dot(user_latent, loc_latent)

        return pred

