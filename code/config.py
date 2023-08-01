
# ROOT = "/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/"
ROOT = "./"
DATA_SOURCE=ROOT + "data/DsDnsPrScTch.csv"
DATA_DEST=ROOT + "processed_data/"
MODEL_DEST=ROOT + "trained_models/"
IMAGE_DEST=ROOT + "model_outputs/figures/"
LIME_DEST=ROOT + "model_outputs/lime/"
SHAP_DEST=ROOT + "model_outputs/shap/"
K_PROJECTS_DEST=IMAGE_DEST + "k_projects/"
ROC_CURVE_DEST=IMAGE_DEST + "roc_curve/"
P_VS_R_CURVE_DEST=IMAGE_DEST + "pr_curve/"

INFO_DEST=ROOT+"model_outputs/info/"

MAX_ROWS=400000

# To label data  
DONATION_PERIOD=30
THRESHOLD_RATIO=0.4

TRAINING_WINDOW = DONATION_PERIOD * 4

MAX_TIME="2015-12-01 00:00:00"
MIN_TIME="2013-01-01 00:00:00"

TEST_SIZE=30
TRAIN_SIZE=TEST_SIZE*6
LEAK_OFFSET=TEST_SIZE*4
WINDOW = TEST_SIZE + TRAIN_SIZE + 2*LEAK_OFFSET


DATE_COLS=["Teacher First Project Posted Date", "Project Fully Funded Date", "Project Expiration Date",
            "Project Posted Date", "Donation Received Date"]
CATEGORICAL_COLS=["Project Type", "Project Subject Category Tree", "Project Subject Subcategory Tree", 
                    "Project Grade Level Category", "Project Resource Category", "School Metro Type",
                    "School State", "School County", "Teacher Prefix", "School Name", "School City", "School District"]

TRAINING_FEATURES=["Project ID", "Project Posted Date", "Project Type", "Project Subject Category Tree", "Project Cost",
                  "Project Subject Subcategory Tree", "Project Grade Level Category", "Project Resource Category", 
                  "School Metro Type", "School Percentage Free Lunch", "School State", "School County",
                  "School Name", "School City", "School District",
                  "Teacher Prefix", "Teacher Project Posted Sequence"]
                #   "Statement Error Ratio", "Title Essay Relativity", "Description Essay Relativity"]

VARIABLES_TO_SCALE=["School Percentage Free Lunch", "Teacher Project Posted Sequence", "Project Cost"]
                    # "Statement Error Ratio", "Title Essay Relativity", "Description Essay Relativity"]