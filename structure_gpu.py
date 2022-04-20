#EXTERNAL LIBRARIES
import cv2
#from numba import jit, cuda


# BUILT-IN LIBRARIES
from enum import Enum
import logging


# CUSTOM LIBRARIES
from structure_data import Structure_Buffer


# Camera Instance Flags
class GPU_FLAGS(Enum):
    CPU = -1
    NVIDIA = 0
    BASLER = 1
    
class GPU_Object():

    def __init__(self, gpu_flag=GPU_FLAGS.CPU, logger_level=logging.INFO):
        self.cuda_enable = False
        self.logger = logging.getLogger("GPU_Object")

        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(name)s : %(message)s',
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)
        
        self.flag_gpu = gpu_flag
        # https://stackoverflow.com/questions/34990228/opencv-gpu-farneback-optical-flow-badly-works-in-multi-threading?rq=1
        """
            GPU module is not thread-safe. It uses some global variables, like __constant__ memory and texture reference API, 
            which can lead to data race if used in multi-threaded environment.
        """
        self.__buffer_gpu = Structure_Buffer(max_limit=1)

        if not self.initialize():
            # TODO: Destroy Object Itself
            self.__del__()
            # del self


    def __del__(self):
        self.logger.error("Destroying GPU Object")
        return self.is_gpu_exist()
        

    def initialize(self):
        result = self.is_gpu_exist()
        if result is True:
            self.logger.info("GPU is Initializing...")
            self.gpu_variable = cv2.cuda_GpuMat()
            return True
        else:
            self.logger.error("GPU is not found: {}".format(result))
            return False

    def is_gpu_exist(self):
        import re#
        self.opencv_cuda_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n') if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
        self.logger.info("GPU CUDA Information: {}".format(self.opencv_cuda_info))
        if self.opencv_cuda_info:
            self.cuda_enable = True
        else:
            self.cuda_enable = False
        return self.cuda_enable
        

    def get_GPU(self):
        return cv2.cuda_GpuMat()
        
        
    def init_undistort(self, old_camera_matrix, dist_coefs, new_camera_matrix, w, h):
        with self.__buffer_gpu:
            self.old_camera_matrix = old_camera_matrix
            self.dist_coefs = dist_coefs
            self.new_camera_matrix = new_camera_matrix

            map1, map2 = cv2.initUndistortRectifyMap(self.old_camera_matrix, self.dist_coefs, None, self.new_camera_matrix, (w,h), cv2.CV_32FC1)

            self.gpu_map1_dist = cv2.cuda_GpuMat(map1)
            self.gpu_map2_dist = cv2.cuda_GpuMat(map2)

        self.logger.info("Undistortion initialized...")


    def upload(self, src_frame):
        with self.__buffer_gpu:
            self.gpu_variable_process = cv2.cuda_GpuMat(src_frame.shape[0], src_frame.shape[1], cv2.CV_8UC3)
            self.gpu_variable.upload(src_frame)


    def download(self):
        return self.gpu_variable.download()
    

    def undistort(self):
        with self.__buffer_gpu:#
            self.gpu_variable = cv2.cuda.remap(
                self.gpu_variable, 
                self.gpu_map1_dist, 
                self.gpu_map2_dist, 
                cv2.INTER_LINEAR
            )

    def warp_Affine(self, M, w, h, get_map=False, flags=cv2.INTER_LINEAR):
        with self.__buffer_gpu:
            if get_map:#
                map1, map2 = cv2.cuda.buildWarpPerspectiveMaps(M, 0, (w,h))
                map1_gpu = cv2.cuda_GpuMat(map1)
                map2_gpu = cv2.cuda_GpuMat(map2)
                self.gpu_variable = cv2.cuda.remap(self.gpu_variable, map1_gpu, map2_gpu, flags=flags)
            
            else:
                self.gpu_variable_process = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
                cv2.cuda.warpAffine(src=self.gpu_variable, dst=self.gpu_variable_process, M=M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                self.gpu_variable = self.gpu_variable_process

    def warp_Perspective(self, M, w, h, get_map=False, flags=cv2.INTER_NEAREST):
        with self.__buffer_gpu:
            if get_map:
                map1, map2 = cv2.cuda.buildWarpPerspectiveMaps(M, 0, (w,h))
                map1_gpu = cv2.cuda_GpuMat(map1)
                map2_gpu = cv2.cuda_GpuMat(map2)

                self.gpu_variable = cv2.cuda.remap(self.gpu_variable, map1_gpu, map2_gpu, flags=flags)

            else:
                self.gpu_variable_process = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
                cv2.cuda.warpPerspective(src=self.gpu_variable, dst=self.gpu_variable_process, M=M, dsize=(w,h), flags=flags)
                self.gpu_variable = self.gpu_variable_process

    def resize(self, w, h, interpolation=cv2.INTER_LINEAR):
        with self.__buffer_gpu:
            self.gpu_variable_process = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
            cv2.cuda.resize(
                src=self.gpu_variable, 
                dst=self.gpu_variable_process, 
                dsize=(w,h), 
                interpolation=interpolation
            )
            self.gpu_variable = self.gpu_variable_process

    def morph_Operations(self, method, kernel, iterations):
        with self.__buffer_gpu:
            morph_filter = cv2.cuda.createMorphologyFilter(op=method, srcType=cv2.CV_8UC1, kernel=kernel, iterations=iterations)
            self.gpu_variable = morph_filter.apply(self.gpu_variable)
            self.kernel = cv2.cuda.createMorphologyFilter(op=method, srcType=cv2.CV_8UC1, kernel=kernel, iterations=iterations)
    
    """def init_gpu_custom_function(self, function, params=None):
        self.gpu_custom_function = function
        self.gpu_custom_function_params = params"""
    
    # function optimized to run on gpu 
    """@jit(target ="cuda")
    def run_gpu_custom_function(self):
        if type(self.gpu_custom_function_params) is list:
            return self.gpu_custom_function(*self.gpu_custom_function_params)
        else:
            return self.gpu_custom_function()"""

    """
    def crop(self, starty, endy, startx, endx):
        h = 
        w = 
        with self.__buffer_gpu:
            self.gpu_variable_process = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
            #print(starty, endy, startx, endx)
            self.gpu_variable.adjustROI(
                startx, endx, 
                starty, endy
            )
    """

