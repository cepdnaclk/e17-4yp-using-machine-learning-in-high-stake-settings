
# ROOT = "/storage/scratch/e17-4yp-xai/Documents/e17-4yp-using-machine-learning-in-high-stake-settings/code/"
ROOT = "/storage/scratch/e17-fyp-xai/projects/e17-4yp-using-machine-learning-in-high-stake-settings/code/"
# ROOT = "./"
# DATA_SOURCE = ROOT + "data/DsDnsPrScTch.csv"
# DATA_SOURCE = ROOT + "data/Dataset.csv"
PATCH = "new/"
DATA_SOURCE = ROOT + "2024/data/FinalDatasetLowFeatures.csv"
DATA_DEST = ROOT + "processed_data/"

ROOT += PATCH
MODEL_DEST = ROOT + "trained_models/"
IMAGE_DEST = ROOT + "model_outputs/figures/"
LIME_DEST = ROOT + "model_outputs/lime/"
SHAP_DEST = ROOT + "model_outputs/shap/"
TREESHAP_DEST = ROOT + "model_outputs/treeshap/"
K_PROJECTS_DEST = IMAGE_DEST + "k_projects/"
ROC_CURVE_DEST = IMAGE_DEST + "roc_curve/"
P_VS_R_CURVE_DEST = IMAGE_DEST + "pr_curve/"
ARTIFACTS_PATH = ROOT + "model_outputs/artifacts/"

INFO_DEST = ROOT+"model_outputs/info/"
# MODEL_RUN_LOG_PATH = INFO_DEST + "model_run_log.py"
PROCESSED_DATA_PATH = DATA_DEST + "processed_final_data_latest.csv"
PROGRAM_LOG_FILE = "program_log.log"

LOAD_PROCESSED_DATA_FLAG = True
REMOVE_SP_CH_FROM_FEATURE_NAMES = False
MAX_ROWS = 500000  # not in use now (18/08/2023)

FIXED_KVAL = 1000

K_START = 1000
K_STEP = 500

# To label data
DONATION_PERIOD = 30
THRESHOLD_RATIO = 0.4
LABEL_PERIOD = 4    # months

TRAINING_WINDOW = DONATION_PERIOD * 4

MAX_TIME = "2017-04-01 00:00:00"
MIN_TIME = "2014-05-01 00:00:00"

TEST_SIZE = 30
TRAIN_SIZE = TEST_SIZE*6
LEAK_OFFSET = TEST_SIZE*4
WINDOW = TEST_SIZE + TRAIN_SIZE + 2*LEAK_OFFSET


DATE_COLS = ["Project Posted Date"]

CATEGORICAL_COLS = ["Project Type", "Project Subject Category Tree", "Project Subject Subcategory Tree",
                    "Project Grade Level Category", "Project Resource Category", "School Metro Type",
                    "School State", "Teacher Prefix", "Resource Vendor Name"
                    ]

TRAINING_FEATURES = ["Project ID", "Project Posted Date", "Project Type", "Project Posted Month", "Project Subject Category Tree", "Project Cost",
                     "Project Subject Subcategory Tree", "Project Grade Level Category", "Project Resource Category",
                     "School Metro Type", "School Percentage Free Lunch", "School State",
                     "Teacher Prefix", "Teacher Project Posted Sequence", "Resource Cost", "Resource Cost Percentage",
                     "Resource Vendor Name"]

VARIABLES_TO_SCALE = ["School Percentage Free Lunch", "Teacher Project Posted Sequence", "Project Cost",  "Project Posted Month",
                      'Teacher Success Rate', 'School City Success Rate', 'School Success Rate',
                      'School County Success Rate', 'Project Count in State',
                      "Resource Cost", "Resource Cost Percentage", "Project Essay Length", 'Project Need Statement Length',
                      'Project Short Description Length']

FINAL_TRAINING_FEATURES = ['Project Cost', 'Project Count in State', 'Project Essay Length', 'Project Grade Level Category',
                           'Project Need Statement Length', 'Project Posted Month', 'Project Resource Category',
                           'Project Short Description Length', 'Project Subject Category Tree', 'Project Subject Subcategory Tree',
                           'Project Type', 'Resource Cost', 'Resource Cost Percentage', 'Resource Vendor Name',
                           'School City Success Rate', 'School County Success Rate', 'School Metro Type',
                           'School Percentage Free Lunch', 'School State', 'School Success Rate', 'Teacher Prefix',
                           'Teacher Project Posted Sequence', 'Teacher Success Rate']
