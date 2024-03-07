from .BaseCompactor import BaseCompactor

class MaxNonActivatedCompactor(BaseCompactor):
    def __init__(self, activated_features):
        # Call the base class's __init__ method to initialize attributes
        super().__init__(None, activated_features)
        
    def compact(self, feature_space):
        if self.activated_features != None:
        
            project_explanation_non_activated_max =[]

            for parent_feature in feature_space.keys():
                if parent_feature in self.activated_features.keys():
                    max_importance = 0
                    for feature, importance in feature_space[parent_feature].items():
                        if feature != self.activated_features[parent_feature] and abs(importance) > abs(max_importance) :
                            max_importance = importance
                            
                    project_explanation_non_activated_max.append([parent_feature,max_importance])
                    
                else:
                    project_explanation_non_activated_max.append([parent_feature,feature_space[parent_feature]])
                    
            return project_explanation_non_activated_max
                    
        else:
            return []