# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
from FlagEmbedding import FlagLLMReranker
import json


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = FlagLLMReranker("./bge-reranker-v2-gemma", use_fp16=True)

    def predict(
        self,
        input_list: str = Input(
            description="JSON string. Can be a list containing one query and one passage pair, or a list of such pairs."
        ),
    ) -> list:
        input_list = json.loads(input_list)
        if all(isinstance(item, list) for item in input_list):
            scores = self.model.compute_score(input_list)
        else:
            scores = self.model.compute_score([input_list])
        return scores
