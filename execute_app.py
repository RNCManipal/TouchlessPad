from json_setting import settings
import os


def execute(labels):  
    funct = ""

    for val in settings.keys():
        if val == labels:
            funct = settings[val]

    if (funct == "Open Notepad"):
        os.system(r"C:\Windows\System32\Notepad.exe")

#    if (funct == "Keyboard"):
#        os.system(r"C:\Program Files (x86)\FreeVK\FreeVK.exe")
#    Need to find a good way to implement keyboard
