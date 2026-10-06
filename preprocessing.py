"""
Root-level alias forwarding to src.preprocessing.
"""
from src.preprocessing import preprocess_training_data, preprocess_image

__all__ = ["preprocess_training_data", "preprocess_image"]
