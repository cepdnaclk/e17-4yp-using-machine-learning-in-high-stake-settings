import numpy as np
from scipy.stats import spearmanr

class Disagreement:
    
    """
    
    The Disagreement class is to facilitate the quantitative comparision of two explanations
    (E.g. feature importance) in the context of machine learning. It provies methods to assess the agreement
    and correlation between two explanaitons, enabling users to gain insights into the similarity or dissimailarity
    between them.
    
    
    Attributes:
        explanation1 (list of tuples): The first explanation, represented as a list of (feature, score) pairs.
        explanation2 (list of tuples): The second explanation, represented as a list of (feature, score) pairs.
        sorted_explanation1 (list of tuples): A sorted version of explanation1 based on the absolute values of scores.
        sorted_explanation2 (list of tuples): A sorted version of explanation2 based on the absolute values of scores.
        feature_ranking_explanation1 (dict): A dictionary mapping features to their ranks in explanation1.
        feature_ranking_explanation2 (dict): A dictionary mapping features to their ranks in explanation2.

    Methods:
        get_feature_agreement(k):
            Calculate the feature agreement between the two explanations for the top k ranked features.

        get_rank_agreement(k):
            Calculate the rank agreement between the two explanations for the top k ranked features.

        get_sign_agreement(k):
            Calculate the sign agreement between the two explanations for the top k ranked features.

        get_signed_rank_agreement(k):
            Calculate the signed rank agreement between the two explanations for the top k ranked features.

        get_rank_correlation(features_F):
            Calculate the Spearman rank correlation between the two explanations for a given set of features.

        get_pairwise_ranking(features_F):
            Calculate the pairwise ranking agreement between the two explanations for a set of features.
    
    """
    
    
    def _intersection_count(self, k: int) -> int:
        """
        Caluclate the intercection count between the top  features of the two explanations

        Args:
            k (int): top k ranked features

        Returns:
            int: intercection count
        """
        # assuming that there is no two same features        
        set_exp1 = set(item[0] for item in self.sorted_explanation1[:k])
        set_exp2 = set(item[0] for item in self.sorted_explanation2[:k])
        
        intersection_count = len(set_exp1.intersection(set_exp2))
        
        
        return intersection_count
    
    
    def get_feature_agreement(self, k:int) -> int:
        """
        Calculate the feature agreement between the two explanations for the top k ranked features.


        Args:
            k (int): top k ranked features

        Returns:
            int: feature agreement
        """
        
        return abs(self._intersection_count(k))/k
    
    def _identical_rank_count(self, k):
        """
        Calculate how many features are in the same rank in the top k features

        Args:
            k (int): top k ranked features

        Returns:
            int: identical count
        """
        identical_count = 0
        
        for item_1, item_2 in zip(self.sorted_explanation1[:k], self.sorted_explanation2[:k]):
            if item_1[0] == item_2[0]:
                identical_count += 1
        
        return identical_count
    
    def get_rank_agreement(self, k):
        """
        Calculate the rank agreement between the two explanations for the top k ranked features.


        Args:
            k (int): top k ranked features

        Returns:
            int: feature rank agreement
        """
        
        return abs(self._identical_rank_count(k))/k
    
    def _identical_sign_count(self, k):
        
        """
        
        Calculate how many interesection elements are with the same sign in the top k features of two explanations


        Args:
            k (int): top k ranked features

        Returns:
            int: identical sign count
        """
        # assuming that there is no two same features
        set_exp1 = set(tuple((item[0], 1 if item[1]>=0 else -1))  for item in self.sorted_explanation1[:k])
        set_exp2 = set(tuple((item[0], 1 if item[1]>=0 else -1))  for item in self.sorted_explanation2[:k])
        
        intersection_count = len(set_exp1.intersection(set_exp2))
        
        return intersection_count
    
    def get_sign_agreement(self, k):
        """
        Calculate the sign agreement between the two explanations for the top k ranked features.


        Args:
            k (int): top k ranked features

        Returns:
            int: sign agreement
        """
        
        return abs(self._identical_sign_count(k))/k
    
    def _identical_rank_sign_count(self, k):
        """
        Calculate how many features are in the same rank and same sign in the top k features

        Args:
            k (int): top k ranked features

        Returns:
            int: identical count
        """
        identical_count = 0
        
        for item_1, item_2 in zip(self.sorted_explanation1[:k], self.sorted_explanation2[:k]):
            if tuple((item_1[0], 1 if item_1[1]>=0 else -1))  == tuple((item_2[0], 1 if item_2[1]>=0 else -1)) :
                identical_count += 1
        
        return identical_count
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    