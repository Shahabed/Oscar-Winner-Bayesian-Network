"""
Bayesian Network for Oscar Winner Prediction
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from typing import Dict


class OscarBayesianNetwork:
    def __init__(self):
        self.model = None
        self.inference = None

    def define_expert_structure(self) -> BayesianNetwork:
        """Define expert knowledge-based structure"""
        edges = [
            ('Genre', 'Other_Awards_Won'),
            ('IMDB_Rating', 'Other_Awards_Won'),
            ('Metacritic', 'Other_Awards_Won'),
            ('Other_Awards_Won', 'Oscar_Nominations'),
            ('Other_Awards_Won', 'Oscar_Wins'),
            ('Oscar_Nominations', 'Oscar_Wins'),
            ('Box_Office', 'Oscar_Nominations'),
            ('Director_Prestige', 'Other_Awards_Won'),
            ('Director_Prestige', 'Oscar_Nominations'),
        ]
        return BayesianNetwork(edges)

    def fit(self, data: pd.DataFrame, use_expert_structure: bool = True):
        if use_expert_structure:
            self.model = self.define_expert_structure()
        else:
            hc = HillClimbSearch(data)
            best_model = hc.estimate(scoring_method=BicScore(data))
            self.model = BayesianNetwork(best_model.edges())

        self.model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
        self.inference = VariableElimination(self.model)
        print("Model fitted successfully.")

    def predict_oscar_wins(self, evidence: Dict):
        if self.inference is None:
            raise ValueError("Model not fitted yet.")

        result = self.inference.query(variables=['Oscar_Wins'], evidence=evidence)
        most_probable = result.values.argmax()
        return {
            'predicted_wins': most_probable,
            'probability': result.values[most_probable],
            'full_distribution': result.values
        }

    def evaluate_model(self, test_data: pd.DataFrame):
        return self.model.log_likelihood(test_data)
