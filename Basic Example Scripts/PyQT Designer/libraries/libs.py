# https://docs.python.org/3/tutorial/modules.html#standard-modules
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)

#sys.path.append("./")

if __name__ == '__main__':
    print("SCRIPT_DIR:", SCRIPT_DIR)
    #print("SCRIPT_DIR dirname:", os.path.dirname(SCRIPT_DIR))
    print("PATH:", sys.path)

