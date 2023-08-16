import pandas as pd
import os
import pickle
from lime_explainer import get_lime_explanation
from kernelshap_explainer import get_shap_explanation
from treeshap_explainer import get_treeshap_explanation
import random

# Function to load the pickle file and return the model
def load_model(pkl_model_path):
  # Load the model
  with open(pkl_model_path,"rb") as file_handle:
    model = pickle.load(file_handle)

  return model

# New pipeline for ruinning the explainations - call this from main
def explanations_pipeline(root, model_paths, train_paths, test_paths, pred_paths, model_name):
  """this pipeline will generate explanations for the given model paths

  Args:
      root (_type_): This is the root path of the model artifacts
      model_paths (_type_): these are the paths to the model artifacts
      train_paths (_type_): these are the paths to the train artifacts
      test_paths (_type_): these are the paths to the test artifacts
      pred_paths (_type_): these are the paths to the prediction artifacts
      model_name (str): this is the model name as a string for identification. ex: "random forest"
      
      all the artifacts should be a list of files and should be in the same order
  """
  assert len(train_paths) == len(test_paths), "There should be same number of train paths and test paths"
  assert len(test_paths) == len(pred_paths), "There should be same number of predictions paths and test paths"
  assert len(model_paths) == len(test_paths), "There should be same number of model_paths paths and test paths"
  
  for i in len(train_paths):
      x_train = pd.read_csv(os.path.join(root, train_paths[i]))
      x_train_cleaned = x_train.drop(["Unnamed: 0", "Project ID", "Label"], axis=1)
      
      fold1 = pd.read_csv(os.path.join(root,test_paths[i]))
      fold_pred =  pd.read_csv(os.path.join(root,  pred_paths[i]))
      Fold1 = pd.concat([fold1, fold_pred["1"]],axis=1)
      Fold1 = Fold1.drop(["Unnamed: 0", "Project ID"],axis=1)
      Fold1_sort = Fold1.sort_values(["1"], ascending=False)
      #Fold1_sort.head()
      x_test = Fold1_sort.drop([ "Label", "1"],axis=1)
      
      top_instance_loc_list = random.sample(range(1000), 50)
      bottom_instance_loc_list = random.sample(range(x_test.shape[0]-1000 , x_test.shape[0]), 50)
      
      
      model = load_model(os.path.join(root, model_paths[i]))

      get_lime_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, ["0", "1"], "classification", model, model_name)
      get_shap_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, model, model_name) 
      get_treeshap_explanation(x_train_cleaned, x_test, top_instance_loc_list, bottom_instance_loc_list, model, model_name) 
  
  return