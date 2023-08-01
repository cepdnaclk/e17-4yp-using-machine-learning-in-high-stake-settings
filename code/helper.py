import pickle

def save_model(path, file_name, model):
    file_path = path + file_name
    pickle.dump(model, file=open(file_path, "wb"))

def load_model(model_file_path):
    return pickle.load(open(model_file_path, 'rb'))