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
    
    
    def get_signed_rank_agreement(self, k):
        """
        Calculate the signed rank agreement between the two explanations for the top k ranked features.


        Args:
            k (int): top k ranked features

        Returns:
            int:  Signed Rank Agreement
        """
        
        return abs(self._identical_rank_sign_count(k))/k
    
    
    
    def get_rank_correlation(self, features: list):
        """
        Calculate the Spearman rank correlation between the two explanations for a given set of features.

        Args:
            features_F (str): user defined features

        Returns:
            float : rank correlation between two explanations for a given set of features
        """
        
        ranking_expanation1 = [self.feature_ranking_explanation1.get(feature) for feature in features_F]
        ranking_expanation2 = [self.feature_ranking_explanation2.get(feature) for feature in features_F]
        
        corr, _ = spearmanr(ranking_expanation1, ranking_expanation2)
        return corr
    
    
    def _relative_ranking(self,exp, fi, fj):
        """
        Compare the relative rankings of two features within an explanation.

        Args:
            exp (dict): A dictionary representing the feature rankings within an explanation.
            fi (str): The name of the first feature to compare.
            fj (str): The name of the second feature to compare.
            
        Returns:
            int: fi is important than fj or not
        """
        
        try:
            if (exp[fi] >= exp[fj]):
                return 1
            else:
                return 0
        except:
            return -1
    
    
    def get_pairwise_rankking(self, features_F):
        """
        Calculate the pairwise ranking agreement between the two explanations for a set of features.

        Args:
            features_F (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        summation = 0
        
        for  i, fi in enumerate(features_F):
            for  j, fj in enumerate(features_F):
                if(i<j):
                    rank1 = self._relative_ranking(self.feature_ranking_explanation1, fi, fj) 
                    rank2 = self._relative_ranking(self.feature_ranking_explanation2, fi, fj) 
                    # print(fi, fj, rank1, rank2)
                    if (i<j and \
                        rank1!=-1 and\
                            rank2!=-1 and\
                                rank1==rank2):
                        summation += 1
                    
        

        return summation/np.math.comb(len(features_F), 2)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    