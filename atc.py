#cython: language_level=3, boundscheck=False
import os
import sys
import libs
import psutil
from os.path import expanduser

"""
home = expanduser("~")
print("home:", home)
app_path = sys.path[0]
print("app_path:", app_path)
"""

if __name__=="__main__":

    # TODO: Remove counter. Useless usage
    counter = 0
    for process in psutil.process_iter():
        #print(home + '/workspace/latest/run.py')
        # ['python3', 'run.py', '-g', '1']
        current_process = process.cmdline()
        if len(current_process) > 1 and current_process[1].__contains__("pano-inspection_run"):
            # **OLDFEATURE** If statement updated with similarity of name
            # == ['python3', home + '/workspace/latest/run.py', '-g', '1']:
            print("Found pano-inspection_run.py!")
            counter += 1

    print("Counter:",counter)
    if counter >= 1:
        print("Program already running...")
    else:
        #os.system('cd ' + home + '/Desktop/lazer-baski-sonrasi-panolarin-goruntu-kontrolu/' + '&& python3 ' + 'pano-inspection_run.py')
        os.system('python3 ' + 'pano-inspection_run.py')
