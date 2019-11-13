#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:01:20 2019

@author: cantaro86
"""

import os
from webbrowser import open_new_tab
from time import sleep
import threading

port_doc = "8888"
port_web = "8888"
name = "Numeric_Finance"
    
UID = os.getuid()
PWD = os.getcwd()
image = "jupyter/scipy-notebook:82d1d0bf0867"
# Alternatives:
#jupyter/scipy-notebook  #"jupyter/tensorflow-notebook"  #jupyter/all-spark-notebook
script = "start-notebook.sh"  


cmd_line = ("docker run --rm -d -p {}:{} \
--name {} \
--user {} \
--group-add users \
-v {}:/home/jovyan/work \
{} {} --NotebookApp.token='' ".format(port_doc, port_web, name, UID, PWD, image, script)  )


thread = threading.Thread( target= os.system(cmd_line) )
thread.start()
thread.join()

print(cmd_line)
sleep(0.8)  # docker is slow to open 
open_new_tab("http://localhost:{}/tree/work".format(port_web))


