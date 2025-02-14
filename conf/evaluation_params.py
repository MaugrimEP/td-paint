"""
parameters only to run the evaluate_predictions.py script
"""
from dataclasses import dataclass


@dataclass
class EvaluationParams:
    # full path to the folder containing the images
    folder_predictions: str = "/folder/to/prediction/default/"
    dataset: str = "not_really_used"
    get_prediction_from: str = "MCG"
    """
    our
    lama
    repaint
    MCG
    MCG_imagenet
    MAT
    """

    diversity_nb: int = 10
