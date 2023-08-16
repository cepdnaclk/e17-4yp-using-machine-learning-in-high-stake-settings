import pandas as pd
import os
import pickle
from lime import get_lime_explanation
from shap import get_shap_explanation
from treeshap import get_treeshap_explanation
import random
# Function to load the pickle file and return the feature importances
def load_model(pkl_model_path):
  # Load the model
  #model = pickle.load(open(pkl_model_path, 'rb'))
    with open(pkl_model_path,"rb") as file_handle:
        model = pickle.load(file_handle)
  # Load the feature importance array
#   importance = model.feature_importances_
  # Get the feature names
#   feat_names = model.feature_names_in_

    return model

#! new pipeline for ruinning the explainations
def explanations_pipeline(root, fold_paths, model_paths, train_paths, test_paths, pred_paths):
    
    assert len(train_paths) == len(test_paths), "There should be same number of train paths and test paths"
    assert len(test_paths) == len(test_paths), "There should be same number of predictions paths and test paths"
    assert len(model_paths) == len(test_paths), "There should be same number of model_paths paths and test paths"
    
    for i in len(fold_paths):
        x_train = pd.read_csv(os.path.join(root, train_paths[i]))
        x_train_cleaned = x_train.drop(["Unnamed: 0", "Project ID", "Label"], axis=1)
        
        fold1 = pd.read_csv(os.path.join(root,test_paths[i]))
        fold_pred =  fold_pred = pd.read_csv(os.path.join(root,  pred_paths[i]))
        Fold1 = pd.concat([fold1, fold_pred["1"]],axis=1)
        Fold1 = Fold1.drop(["Unnamed: 0", "Project ID"],axis=1)
        Fold1_sort = Fold1.sort_values(["1"], ascending=False)
        Fold1_sort.head()
        x_test = Fold1_sort.drop([ "Label", "1"],axis=1)
        
        top_instance_loc_list = random.sample(range(1000), 50)
        bottom_instance_loc_list = random.sample(range(x_test.shape[0]-1000 , x_test.shape[0]), 50)
        
        
        model = load_model(os.path.join(root, model_paths[i]))

        get_lime_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, ["0", "1"], "classification", model, "random_forest")
        get_shap_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, ["0", "1"], "classification", model, "random_forest") 
        get_treeshap_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, ["0", "1"], "classification", model, "random_forest") 
    
    return