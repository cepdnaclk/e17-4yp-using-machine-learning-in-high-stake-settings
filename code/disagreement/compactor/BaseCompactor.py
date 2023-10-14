
from abc import ABC, abstractmethod


class BaseCompactor(ABC):
    
    """
    This class defines a base compactor for feature space compaction.
    Subclasses should implement the 'compact' method to perform compaction.    
    
    """
    
    
    def __init__(self, categorical_columns=None, activated_features =None):
        """
        Initializes the compactor with optional 'categorical_columns' and 'activated_features'.

        Args:
            categorical_columns: List of categorical columns (optional).
            activated_features: List of activated features (optional).
        """
        self.categorical_columns = categorical_columns
        self.activated_features = activated_features
    @abstractmethod   
    def compact(self, feature_space):
        """
        This method should be implemented by subclasses to perform feature space compaction.

        Args:
            feature_space (dict): The feature space to be compacted.

        Returns:
            list: A list of lists containing features and their respective feature importance.

        Example:
        input:
        {
            'Project Cost': -0.02602504681196933,
            'School State': {
                'School State_Illinois': -0.02180424855480396,
                'School State_New York': -0.0015074806829943554,
                'School State_New Jersey': -0.0005795116676927136,
                'School State_Massachusetts': 0.0005370641225545114,
                'School State_Pennsylvania': 0.0005351177916600136
            }
        }

        output:
        [
            ['Project Cost', -0.02602504681196933],
            ['School State', 0.0005351177916600136]
        ]
        """
        pass
    
        
        
        
        
        
        
       
        
        