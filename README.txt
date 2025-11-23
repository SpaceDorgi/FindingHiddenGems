DESCRIPTION
This package has 2 components. The first being the recommendation system portion, where you can process the yelp dataset to create custom features to train and evaluate an attribute aware latent factor model.
This also includes an example usage directory with an example usage notebook of the building of our end to end system. From processing and filtering the data to training and evaluating a model to making recommendations.
This part of the package implements a recommendation system utilizing a form of matrix factorization to provide recommendations for Yelp restaurants to users.
It combines content-based features and collaborative filtering (attribute aware latent factor model or collaborative filtering with side information) to generate personalized restaurant recommendations.
The codebase includes data preprocessing, feature extraction, model training, evaluation, plotting utilities, and example Jupyter notebooks.

The 2nd main component to this package is the visualization portion, where you can visualize a users recommendations that have been reranked by the restaurant hidden gem score.
This piece is implemented as an single page application (SPA) that is deployed on AWS Lambda and can be accessed via a public URL. Access the visualization app here:https://2ucwrakk2sffs447greh5cbnte0jjtdr.lambda-url.us-east-2.on.aws/ (it could take a little time to spin up, try to click the user dropdown and select a user to find their recommendations)
The app is built using Flask as the backend framework, with HTML, CSS, and JavaScript for the frontend.
The HTML/JavaScript frontend consumes the static precomputed recommendations, providing users with an interactive, responsive experience on the interactive map.
This serverless deployment model leverages AWS Lambda's scalability and cost efficiency for serving the application and delivering a continuously refreshed set of static (for now), yet highly personalized, recommendations to end-users.


INSTALLATION
Requirements: Python 3.10+

1. Download the yelp dataset :
   - dataset url: https://business.yelp.com/data/resources/open-dataset/
   - set it in the project directory of your choice

2. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`

3. Install dependencies or the repo:
   - repo url: https://github.com/SpaceDorgi/FindingHiddenGems
   - clone repo: https://github.com/SpaceDorgi/FindingHiddenGems.git
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`


EXECUTION (Satya, I will need some help here, and we will likely need to update it)

A) Example usage for building a model to compute recommendations:
1. Building a demo model from scratch using the notebooks:
   - Go to the example usage directory
   - Open the example_usage_notebook.ipynb notebook
    - Update the file paths to point to your local dataset location or use the subsets provided in the example usage directory
   - Run the notebook cells in order to process the data, extract features, train the model, evaluate it, and generate recommendations.
   - Note this notebook was used on a smaller subset of the data for demonstration purposes. For full scale training, refer to the detailed notebooks below.


2. For a more detailed look:
    - Look at notebooks 1-7 in the notebooks/ directory that build the recommendation system step by step.
    - Run the notebooks in sequence: notebooks/1-data_exploration.ipynb -> notebooks/2-location_classifier.ipynb -> notebooks/3-apply_vader_to_reviews.ipynb -> notebooks/4-update_jsons_with_new_data.ipynb -> notebooks/5-final_model.ipynb -> notebooks/6-compare_models.ipynb -> 7-load_and_use_model.ipynb.
    - Each notebook incrementally builds the pipeline: data exploration and cleaning, location/attribute classifiers, sentiment features, creating new features, final model training/evaluation, and model comparison.


3.  What each notebook provides
    1-data_exploration.ipynb: dataset overview, filtering to restaurants
    2-location_classifier.ipynb: extracts users most reviewed in city and assigns that as the users home city
    3-apply_vader_to_reviews.ipynb: extracts sentiment features from reviews
    4-update_jsons_with_new_data.ipynb: adds new features to the dataset that are used downstream
    5-final_model.ipynb: assembles feature matrices, trains the recommender(s), runs evaluation and plots
    6-compare_models.ipynb: compares different model variants and configurations — helps us test our model against others.
    7-load_and_use_model.ipynb: shows how to load a saved model and use it to generate recommendations for a user.


B) Visualization app deployment and usage:
1. Prepare data and calculate HG score: Visualization\datapreparationforHGScore

FULL_DB_PATH = os.path.join(SCRIPT_DIR, "data/yelp_all_fromcsv.db")
TARGET_USERS_FILE = os.path.join(SCRIPT_DIR, "data/users.txt")
SMALL_DB_PATH = os.path.join(SCRIPT_DIR, "data/filtered_yelp_subset.db")


	Script							Required Input Files							Output Files
	step0_insertCSVDataTosqlite.py	yelp_restaurants_filtered_with_new_features.csv,yelp_restaurant_reviews_with_vader.csv,yelp_users_with_taste_profile.csv	yelp_all_fromcsv.db
	step1_sqlNewdbAllBus.py			yelp_all_fromcsv.db, users.txt					filtered_yelp_subset.db
	step2_calculateHGScore.py		filtered_yelp_subset.db, business_ids.txt		business_metrics.json (or parquet/csv)
	step3_MergeHG.py				user_top25_recs.csv, business_metrics.json		user_recs_25_with_gem_score.csv


	business_ids.txt = contains all unique business ids in a text file \n separated.
	users.txt	= contains all unique user ids in a text file \n separated.

2)	Visualization UI: Visualization\ToLambda
	Contains sample data, .html and app.py