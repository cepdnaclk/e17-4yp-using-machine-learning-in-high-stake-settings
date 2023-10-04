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
    
    
    def __init__(self, explanation1, explanation2):
        #explanations
        self.explanation1 = explanation1
        self.explanation2 = explanation2
        #explanations are sorted based on the absolute values
        self.sorted_explanation1 = sorted(self.explanation1, key= lambda x: abs(x[1]), reverse=True)
        self.sorted_explanation2 = sorted(self.explanation2, key= lambda x: abs(x[1]),reverse=True)
        # ranks of each feature based on the absolute value
        self.feature_ranking_explanation1 = {}
        self.feature_ranking_explanation2 = {}
        
        for rank , (feature, _) in enumerate(self.sorted_explanation1):
            self.feature_ranking_explanation1[feature] = rank
            
        for rank , (feature, _) in enumerate(self.sorted_explanation2):
            self.feature_ranking_explanation2[feature] = rank
            
        feature_space_with_importance_explanation1 = self._feature_space_with_feature_importance(explanation1)
        feature_space_with_importance_explanation2 = self._feature_space_with_feature_importance(explanation2)
        


        

    def _feature_space_with_feature_importance(project_explanation:  dict) -> dict:
        feature_space_with_importance = {}
        for feature , important_score in  project_explanation:
            if len(feature.split("_")) == 1:
                feature_space_with_importance[feature.split("_")[0]] = important_score

            elif feature.split("_")[0] in feature_space_with_importance.keys():
                feature_space_with_importance[feature.split("_")[0]][feature]=  important_score
            else:
                feature_space_with_importance[feature.split("_")[0]]= {}
                feature_space_with_importance[feature.split("_")[0]][feature]=  important_score
        return feature_space_with_importance
        
    
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
    
    
    
    def get_rank_correlation(self, features_F: list):
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
    
    def get_disagreement(self, k:int, features_F:list)-> list:

        # feature agreement
        feature_agreement = self.get_feature_agreement(k)

        # rank agreement
        rank_agreement = self.get_rank_agreement(k)

        # Test sign agreement
        sign_agreement = self.get_sign_agreement(k)

        #  signed rank agreement
        signed_rank_agreement = self.get_signed_rank_agreement(k)
        
        
        if features_F != None:
            #  rank correlation
            rank_correlation = self.get_rank_correlation(features_F)

            #  pairwise ranking
            pairwise_ranking = self.get_pairwise_rankking(features_F)
            
            return { "feature_agreement" : feature_agreement, "rank_agreement": rank_agreement, "sign_agreement" :sign_agreement, "signed_rank_agreement" : signed_rank_agreement, "rank_correlation" : rank_correlation,"pairwise_ranking" : pairwise_ranking}
        
        else:
            return { "feature_agreement" : feature_agreement, "rank_agreement": rank_agreement, "sign_agreement" :sign_agreement, "signed_rank_agreement" : signed_rank_agreement}

    def print_disagreement(self, k:int, features_F:list) -> None:

        feature_agreement = self.get_feature_agreement(k)
        print(f"Feature Agreement: {feature_agreement}")

        rank_agreement = self.get_rank_agreement(k)
        print(f"Rank Agreement: {rank_agreement}")

        sign_agreement = self.get_sign_agreement(k)
        print(f"Sign Agreement: {sign_agreement}")

        signed_rank_agreement = self.get_signed_rank_agreement(k)
        print(f"Signed Rank Agreement: {signed_rank_agreement}")
        
        if features_F != None:

            rank_correlation = self.get_rank_correlation(features_F)
            print(f"Rank Correlation: {rank_correlation}")

            pairwise_ranking = self.get_pairwise_rankking(features_F)
            print(f"Pairwise Ranking: {pairwise_ranking}")
        
    
    
            
if __name__ == "__main__":
    
    # Sample explanations (you can replace these with your actual explanations)
    explanation1 = [['Feature1', 0.85], ['Feature2', 0.72], ['Feature3', 0.68], ['Feature4', 0.53], ['Feature5', 0.42]]
    explanation2 = [['Feature1', 0.75], ['Feature2', -0.82], ['Feature3', 0.63], ['Feature4', 0.57], ['Feature5', 0.49]]

    # Sample set of features
    features_F = ['Feature1', 'Feature2', 'Feature3']

    # Create an instance of the Disagreement class
    disagreement_calculator = Disagreement(explanation1, explanation2)

    # Test various methods
    k = len(features_F)

    # Test feature agreement
    feature_agreement = disagreement_calculator.get_feature_agreement(k)
    print(f"Feature Agreement: {feature_agreement}")

    # Test rank agreement
    rank_agreement = disagreement_calculator.get_rank_agreement(k)
    print(f"Rank Agreement: {rank_agreement}")

    # Test sign agreement
    sign_agreement = disagreement_calculator.get_sign_agreement(k)
    print(f"Sign Agreement: {sign_agreement}")

    # Test signed rank agreement
    signed_rank_agreement = disagreement_calculator.get_signed_rank_agreement(k)
    print(f"Signed Rank Agreement: {signed_rank_agreement}")

    # Test rank correlation
    rank_correlation = disagreement_calculator.get_rank_correlation(features_F)
    print(f"Rank Correlation: {rank_correlation}")

    # Test pairwise ranking
    pairwise_ranking = disagreement_calculator.get_pairwise_rankking(features_F)
    print(f"Pairwise Ranking: {pairwise_ranking}")

    
    
    
    
    
    
        
    
    
    
    





    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
