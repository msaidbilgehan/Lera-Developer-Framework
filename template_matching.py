#cython: language_level=3, boundscheck=False
import enum
import math

import cv2
import numpy as np
from ordered_enum import OrderedEnum

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


class APPLIABLE_ALGORITHMS(OrderedEnum):
    SQUARED_DIFFERENCE = 0
    NORMALIZED_SQUARED_DIFFERENCE = 1
    CROSS_CORRELATION = 2
    NORMALIZED_CROSS_CORRELATION = 3
    COSINE_COEFFICIENT = 4
    NORMALIZED_COSINE_COEFFICIENT = 5
    BASIC_CROSS_CORRELATION = 6
    XOR_OPERATOR = 7
    

class MATCHING:
    def __init__(self):
        self.interpolation_range_list = np.array([0,0,1], dtype=np.float32)
        self.blank_list = np.array([], dtype=np.float32)
        
    def compute(self, reference_image, sample_image, algorithm=APPLIABLE_ALGORITHMS.BASIC_CROSS_CORRELATION):
        if algorithm.value == 6:
            refImage = np.longlong(reference_image)
            sampleImage = np.longlong(sample_image)

            conv = refImage * sampleImage
            sumof_conv = np.longlong(np.sum(conv))

            squareof_refImage = refImage * refImage
            squareof_sampleImage = sampleImage * sampleImage

            sumof_squareof_refImage = np.longlong(np.sum(squareof_refImage))
            sumof_squareof_sampleImage = np.longlong(np.sum(squareof_sampleImage))

            multipleof_matching = sumof_squareof_refImage * sumof_squareof_sampleImage
        
            if multipleof_matching <= 0:
                similarity_match_ratio = 0
                return similarity_match_ratio

            squarerootof_multipleof_matching = math.sqrt(multipleof_matching)

            if (sumof_conv >= 0) and (squarerootof_multipleof_matching > 0):
                similarity_match_ratio = sumof_conv / squarerootof_multipleof_matching
                                
                return similarity_match_ratio
            
            
        else:
            result_of_method = cv2.matchTemplate(sample_image, reference_image, method=algorithm.value)
            
            self.interpolation_range_list[1] = result_of_method[0][0]
            
            similarity_match_ratio = cv2.normalize(self.interpolation_range_list, self.blank_list, 0, 1, cv2.NORM_MINMAX)
            
            return similarity_match_ratio[1]