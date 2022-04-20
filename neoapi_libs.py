# https://docs.python.org/3/tutorial/modules.html#standard-modules
from platform import system
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)

os_name = system()

if os_name == "Linux":
    sys.path.append("third_party_imports/neoapi/linux/neoapi")
elif os_name == "Windows":
    sys.path.append("third_party_imports/neoapi/windows/neoapi")
else:
    print(f"No NEOAPI implementation for Operating System: {os_name}")

#sys.path.append("./")

if __name__ == '__main__':
    print("SCRIPT_DIR:", SCRIPT_DIR)
    #print("SCRIPT_DIR dirname:", os.path.dirname(SCRIPT_DIR))
    print("PATH:", sys.path)

