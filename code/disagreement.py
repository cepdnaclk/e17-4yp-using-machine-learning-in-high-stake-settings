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
    
    