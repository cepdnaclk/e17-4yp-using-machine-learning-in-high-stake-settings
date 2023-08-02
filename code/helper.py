import pickle
import config
import os

def save_model(path, file_name, model):
    file_path = path + file_name
    pickle.dump(model, file=open(file_path, "wb"))

def load_model(model_file_path):
    return pickle.load(open(model_file_path, 'rb'))

def create_dirs(models=None):
    if not models:
        models = ['decision_tree', 'log_reg', 'random_forest', 'svm']
    paths = [
        config.ARTIFACTS_PATH+model_dir+'/' for model_dir in models
    ] + [
        config.IMAGE_DEST+model_dir+'/' for model_dir in models
    ] + [
        config.IMAGE_DEST+'k_projects/'+model_dir+'/' for model_dir in models
    ] + [
        config.INFO_DEST+model_dir+'/' for model_dir in models
    ] + [
        config.ROOT+'trained_models/', config.ROOT+'processed_data/'
    ]
    print(len(paths))
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            
    print('Created all directories!')
