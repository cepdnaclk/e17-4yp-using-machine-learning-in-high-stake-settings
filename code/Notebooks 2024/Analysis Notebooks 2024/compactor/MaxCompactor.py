from .BaseCompactor import BaseCompactor


class MaxCompactor(BaseCompactor):
    def __init__(self, categorical_columns):
        # Call the base class's __init__ method to initialize attributes
        super().__init__(categorical_columns, None)
    
    
    def compact(self, feature_space):
        if self.categorical_columns != None:
            compacted_feature_abs_max = []

            for parent_feature, children_space in feature_space.items():
                if parent_feature in self.categorical_columns:
                    compacted_feature_abs_max.append([parent_feature,  max(children_space.values(), key= lambda x : abs(x))])
                else:
                    compacted_feature_abs_max.append([parent_feature, feature_space[parent_feature]])

            return compacted_feature_abs_max
        else:
            return []