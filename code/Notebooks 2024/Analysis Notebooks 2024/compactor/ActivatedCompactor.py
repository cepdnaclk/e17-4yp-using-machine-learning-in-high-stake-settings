from .BaseCompactor import BaseCompactor

class ActivatedCompactor(BaseCompactor):
    def __init__(self, activated_features):
        # Call the base class's __init__ method to initialize attributes
        super().__init__(None, activated_features)
        
    def compact(self, feature_space):
        if self.activated_features != None:
        
            compacted_project_explanation_activated_importance =[]

            for parent_feature in feature_space.keys():
                if parent_feature in self.activated_features.keys():
                    compacted_project_explanation_activated_importance.append([parent_feature,feature_space[parent_feature][self.activated_features[parent_feature]]])
                else:
                    compacted_project_explanation_activated_importance.append([parent_feature,feature_space[parent_feature]])

            return compacted_project_explanation_activated_importance
        
        else:
            return []