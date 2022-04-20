# https://docs.python.org/3/tutorial/modules.html#standard-modules
# https://docs.python.org/3/library/os.path.html#os.path.expanduser
import sys
from platform import system
from os.path import expanduser
import os


os_name = system()
home = expanduser("~").replace("\\", "/")

run_path = os.getcwd().replace("\\", "/")
app_path = sys.path[0]


"""
sys.path.append("lera-developer-framework")
"""
sys.path.append(home + "/Workspace/lera-developer-framework")

print("OS Name:", os_name)
#print("HOME PATH:", home)
#print("RUN PATH:", run_path)
#print("APP PATH:", app_path)
#print("PATH Variables:", sys.path)
