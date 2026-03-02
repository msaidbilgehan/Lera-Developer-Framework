import enum
import math

import numpy as np
from ordered_enum import OrderedEnum

import Levenshtein as lev


class APPLICABLE_ALGORITHMS(OrderedEnum):
    Levenshtein = 0
    Jaro_Winkler = 1
    

class STRING_SIMILARITY:
    def __init__(self):
        self.interpolation_range_list = np.array([0,0,1], dtype=np.float32)
        self.blank_list = np.array([], dtype=np.float32)
        
    def compute(self, reference_string, sample_string, algorithm=APPLICABLE_ALGORITHMS.Levenshtein):
            
        reference_string = str(reference_string)
        sample_string = str(sample_string)
        
        if algorithm.value == 0:
            distance = lev.distance(reference_string, sample_string)
            max_len = max(len(reference_string), len(sample_string))
            similarity_match_ratio = 1 - distance / max_len
            
        elif algorithm.value == 1:
            similarity_match_ratio = lev.jaro_winkler(reference_string, sample_string)
        
        return similarity_match_ratio