import os
import psutil
from os.path import expanduser


home = expanduser("~")

starter_command = "python3"
app_path = "/some/latest/"
app_command = "pcb_component_inspection.py"
app_parameters = "-g 1"

full_command = starter_command + " " + home + app_path + app_command + " " + app_parameters

if __name__=="__main__":

    # TODO: Remove counter. Useless usage
    is_running = False
    for process in psutil.process_iter():
        #print(home + '/workspace/latest/run.py')
        # ['python3', 'run.py', '-g', '1']
        current_process = process.cmdline()
        if len(current_process) > 1 and current_process[1].__contains__(app_command):
            # **OLDFEATURE** If statement updated with similarity of name
            # == ['python3', home + '/workspace/latest/run.py', '-g', '1']:
            print("Found pcb_component_inspection.py!")
            is_running = True

    if is_running:
        print("Program already running...")
    else:
        os.system(full_command)
