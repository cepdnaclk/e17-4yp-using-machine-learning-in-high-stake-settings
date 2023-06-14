DATA_SOURCE="/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/data/DsDnsPrScTch.csv"
DATA_DEST="/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/processed_data/"
MODEL_DEST="/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/trained_models/"
IMAGE_DEST="/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/model_outputs/figures/"

MAX_ROWS=10000
TOTAL_TIME_PERIOD_DAYS=120
EVAL_PERIOD_DAYS=30
TRAINING_WINDOW = EVAL_PERIOD_DAYS * 4
MAX_TIME="2015-12-01 00:00:00"
MIN_TIME="2013-01-01 00:00:00"

# To label data  
DONATION_PERIOD=30
THRESHOLD_RATIO=0.4

DATE_COLS=["Teacher First Project Posted Date", "Project Fully Funded Date", "Project Expiration Date",
            "Project Posted Date", "Donation Received Date"]
CATEGORICAL_COLS=["Project Type", "Project Subject Category Tree", "Project Subject Subcategory Tree", 
                    "Project Grade Level Category", "Project Resource Category", "School Metro Type",
                    "School State", "School County", "Teacher Prefix"]
TRAINING_FEATURES=["Project ID", "Project Type", "Project Subject Category Tree", 
                  "Project Subject Subcategory Tree", "Project Grade Level Category", 
                  "Project Resource Category", "School Metro Type", "School Percentage Free Lunch", 
                  "School State", "School County", "Teacher Prefix", "Statement Error Ratio", 
                  "Title Essay Relativity", "Description Essay Relativity"]
VARIABLES_TO_SCALE=["School Percentage Free Lunch", "Statement Error Ratio", 
                    "Title Essay Relativity", "Description Essay Relativity"]