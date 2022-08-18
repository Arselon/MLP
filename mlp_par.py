# HTMS_AI v.1.0
# MLP1
# © A.S.Aliev, 2022
# 

import os
import sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    # root MLP software directory (on local computer)

FILES_DIR = os.path.join(BASE_DIR, "data/files") 
    # root directory for trainig/validation files (on local computer)

LOCAL_DB_DIR = os.path.join(BASE_DIR, "data") 
    # root directory for hypertables on local computer

DB_URL = "127.0.0.1:3570"  # server IP and main port of the Cageserver

DB_ROOT = "HTMS_MLP"   
    # additional directory for hypertables
    # full path will be: 
            # on server - server root directory for hypertables\\DB_ROOT 
            # on local computer - BASE_DIR\\LOCAL_DB_DIR\\DB_ROOT

LOG_ERR =os.path.join(BASE_DIR, "logs/cage_error.log")  
    # default errlog file

LOG_PRINT =os.path.join(BASE_DIR, "logs/cage_print.txt")  
    # file for copying/redirecting system output

JWT_ADMIN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJidXNpbmVzc19vbiIsInVzZXJfbmFtZSI6ImNvd29ya2VyIiwiaWF0IjoxNjEyMTAwNDE0LCJleHAiOjE2NDM2MzY0MTQsInBlcm1pc3Npb24iOiJhZG1pbiIsImZvbGRlciI6W10sInNpemUiOi0xfQ._Tz3kYLrSTwzzFT9XPuL1gmJhV_MN3g4rsg-23f4WZ0"
    # JWT token for authorizing on Cageserver

# logistic function
def logistic(x):
    return 1.0/(1 + math.exp(-x))

# the derivative of the logistic function
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))