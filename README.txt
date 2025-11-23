DESCRIPTION
This package has 2 components. The first being the recommendation system portion, where you can process the yelp dataset to create custom features to train and evaluate an attribute aware latent factor model.
This also includes an example usage directory with an example usage notebook of the building of our end to end system. From processing and filtering the data to training and evaluating a model to making recommendations.
This part of the package implements a recommendation system utilizing a form of matrix factorization to provide recommendations for Yelp restaurants to users.
It combines content-based features and collaborative filtering (attribute aware latent factor model or collaborative filtering with side information) to generate personalized restaurant recommendations.
The codebase includes data preprocessing, feature extraction, model training, evaluation, plotting utilities, and example Jupyter notebooks.

The 2nd main component to this package is the visualization portion, where you can visualize a users recommendations that have been reranked by the restaurant hidden gem score.
This piece is implemented as a single page application (SPA) that is deployed on AWS Lambda and can be accessed via a public URL. Access the visualization app here:https://2ucwrakk2sffs447greh5cbnte0jjtdr.lambda-url.us-east-2.on.aws/ (it could take a little time to spin up, try to click the user dropdown and select a user to find their recommendations)
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


EXECUTION

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

	Food Recommendation application is architected as a Serverless Single-Page Application (SPA), utilizing a Function-as-a-Service (FaaS) model for maximum scalability and minimal infrastructure management.
	Deployment Environment: AWS Lambda.
	Web Framework: Flask (app.py) serves as the core micro-framework, packaged within the Lambda execution environment. It is responsible for handling routing and serving the static assets and computed data.
	Data Persistence & Retrieval: Pre-computed recommendation vectors, generated by an offline process, are stored in CSV format. This data is typically uploaded to and retrieved from an AWS Simple Storage Service (S3) bucket.
	
	Visualization Architecture Flow:
	The client (browser) makes a request to the AWS Lambda Function URL.
	The API Gateway triggers the configured AWS Lambda function (containing the Flask application and necessary dependencies in Lambda Layers).
	The Lambda function's handler loads the static HTML/JavaScript (leaflet.js) assets and the CSV recommendation data into memory.
	The Flask application serves the SPA components to the client.
	The client-side JavaScript processes the CSV data to render the personalized recommendations in the user interface. This shifts the computational burden for rendering from the server to the client.
	This static-serving pattern over FaaS effectively minimizes the per-request latency and cost, as the Lambda function primarily serves read-only data and static files, with the intensive recommendation computation being performed off-line.
	
	Key Technical Aspects:
	Serverless Architecture: Deployment on AWS Lambda with Function URL, optimizing for low operational overhead and dynamic scaling.
	Recommendation Computation: An offline process (implied by "static computed recommendations") generating the results, which are then persisted as flat-file data (CSV).
	Web Framework: Flask is used to bootstrap the application, handling asset delivery and potentially serving the pre-computed CSV data to the frontend JavaScript for client-side rendering.
	Data Format: CSV is the primary format for storing and transmitting the final recommendation vectors/matrices.

	Steps for getting final data:
	
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
	
3)	Create a Lambda function in AWS account.
	zip the "Visualization\ToLambda" and upload it to lambda/code.
	Create a function URL in configurations tab.
	Deploy the Lambda function.
	Current access to the visualization app here: https://2ucwrakk2sffs447greh5cbnte0jjtdr.lambda-url.us-east-2.on.aws/
	
