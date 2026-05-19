import json
import os
from typing import Optional
import numpy as np
from torch.utils.data import WeightedRandomSampler

from super_gradients.common.registry import register_sampler

@register_sampler("ChessHardExampleSampler")
class ChessHardExampleSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset=None,
        hard_factors_file: Optional[str] = None,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        """
        Sampler that oversamples specific hard examples based on a JSON list of weights.
        
        :param dataset: The dataset object. Needed to determine fallback length if file is missing.
        :param hard_factors_file: Path to a JSON file containing a list of floats (weights per image).
        :param num_samples: Number of samples to draw. Defaults to len(weights).
        """
        
        if hard_factors_file is None or not os.path.exists(hard_factors_file):
            print(f"ChessHardExampleSampler: {hard_factors_file} not found. Falling back to uniform sampling.")
            if dataset is None:
                raise ValueError("dataset must be provided if hard_factors_file is missing.")
            weights = np.ones(len(dataset))
        else:
            with open(hard_factors_file, "r") as f:
                factors = json.load(f)
                
            if isinstance(factors, list):
                weights = np.array(factors)
                if dataset is not None and len(weights) != len(dataset):
                    print(f"Warning: weights length ({len(weights)}) does not match dataset length ({len(dataset)}).")
            else:
                raise ValueError("hard_factors_file should contain a JSON list of weights.")
        
        # normalize
        weights = weights / weights.sum()
        
        super().__init__(weights=weights, num_samples=num_samples or len(weights), replacement=True, generator=generator)
