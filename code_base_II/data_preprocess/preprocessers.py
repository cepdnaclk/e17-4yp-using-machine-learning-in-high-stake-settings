import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



# Preprocessing Strategy
class Preprocessor:
    def preprocess(self, data):
        raise NotImplementedError

class NumericPreprocessor(Preprocessor):
    def preprocess(self, data):
        numeric_features = ['teacher_number_of_previously_posted_projects']
        numeric_transformer = StandardScaler()
        ct = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features)],
                               remainder='passthrough')
        preprocessed_data = ct.fit_transform(data)
        return pd.DataFrame(preprocessed_data, columns=data.columns)

class CategoricalPreprocessor(Preprocessor):
    def preprocess(self, data):
        categorical_features = ['school_state', 'teacher_prefix', 'project_grade_category']
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)
        ct = ColumnTransformer(transformers=[('categorical', categorical_transformer, categorical_features)],
                               remainder='passthrough')
        preprocessed_data = ct.fit_transform(data)
        return pd.DataFrame(preprocessed_data, columns=ct.get_feature_names_out())
