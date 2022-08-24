# HTMS_AI v.1.0
# Script for start
# © A.S.Aliev, 2022
# 

import os
import sys
import os
import time
import datetime
import copy

from cage_api import *
from cage_api.cage_err import Logger

from mlp_par import *
from mlp1 import load_train_valid, mlp_train_valid, mlp_load_exec


#----------------------------------------------------------------------------------------

if __name__ == "__main__":

    Log_err = "htms_error.log"  # default errlog file   
    Log_print = "htms_print.txt"  # file for copying/redirecting system

    LR = input("\n ??? Start MLP (y/n) - (Yes by default) >")
    if LR == "" or LR[0] == "y" or LR[0] == "Y":
        errlog.write(
                "\n\n"
                + datetime.datetime.now().strftime(
                    "%d.%m.%y  %H:%M:%S. %f  -------   NEW MLP SESSION  -------- "
                )
            )
        errlog.flush()
        """
        sys_out= ("standard",)      # without print to console
        q5= input ('\n Keep parallel stdout to console   (y/n - "Yes" by default) >')
        if q5=='' or q5[0]=='y' or q5[0]=='Y':
            sys_out= ("custom",) 

        terminal = True
        sys.stdout = Logger(term=terminal)
        """

        files_dir= input ('\n Input full path to files directory   ("%s" by default) - >'% \
            FILES_DIR)      
        if files_dir == '':
            files_dir = FILES_DIR

        q0 = input("\n    ??????    Test without Cage server (y/n) >")
        if q0 == "" or q0[0] == "y" or q0[0] == "Y":
            db_url = ""
            local_root= input ('\n Input full path to local hypertables directory - ("%s" by default) - >'% \
                LOCAL_DB_DIR)      
            if local_root == '':
                local_root = LOCAL_DB_DIR
        else:
            local_root = ""
            db_url= input ('\n Input HTMS server URL - ("%s" by default) - >'% \
                DB_URL)      
            if db_url == '':
                db_url = DB_URL
        
        q1= input ('\n Redirect stdout to file (y/n) - (Yes by default) >')
        if q1=='' or q1[0]=='y' or q1[0]=='Y':
            q2= input ('\n Input file name for stdout copy - ("%s" by default) >'% Log_print)
            q3= input ('\n Keep parallel stdout (y/n) - ("Yes by default) >')
            if q3=='' or q3[0]=='y' or q3[0]=='Y':
                terminal=True
            else: 
                terminal=False
            if q2 =='':
                sys.stdout = Logger(term=terminal)
            else: 
                sys.stdout = Logger(filename=q4, term=terminal)

        var= input ('\n\n    ??????    Training (by default) or execution (any letter) - >')
        if var == '':

            mlp_ver= input ('\n Input MLP version - (0 by default) - >')
            if mlp_ver == '':
                mlp_ver = 0
            else:
                mlp_ver = int (mlp_ver)            

            if mlp_ver==1:
                mlp1_RAM.mlp_1_RAM(files_dir,)

            elif mlp_ver==2:
                mlp2_RAM.mlp_2_RAM(files_dir,)
             
            elif mlp_ver==0:
                mlp_db_name='mlp1'
                q6=input("\n  MLP HTMS hypertable name - ('%s' by default) >"%
                         mlp_db_name)
                if q6!='':
                    mlp_db_name=q6

                train_db_name='train1'
                q8=input("\n Training & validation HTMS hypertable name - ('%s' by default) >"%
                             train_db_name)
                if q8!='':
                        train_db_name=q8

                q4 = input("\n    ??????    Create database (y/n) - (No by default) >")
                if q4 == "" or q4[0] == "n" or q4[0] == "N":
                    pass
                else:
                    train_excel_file= 'Train_MLP_Exp_3'
                    valid_excel_file= 'Valid_MLP_Exp_3_norandom'

                    q5=input("\n   Training excel file name - ('%s' by default) >"%
                             train_excel_file)
                    if q5!='':
                        train_excel_file=q5
                    training_data_root= os.path.join(files_dir, train_excel_file+'.xlsx')

                    q7=input("\n    Validation excel file name - ('%s' by default) >"%
                             valid_excel_file)
                    if q7!='':
                        valid_excel_file=q5
                    validation_data_root= (os.path.join(files_dir, valid_excel_file+'.xlsx'))

                    load_train_valid.load_mlp_db (files_dir, db_url, DB_ROOT,
                             training_data_root, validation_data_root,
                             mlp_db_name, train_db_name, local_root=local_root)
                
                while (True):                    
                    mlp_train_valid.mlp_train_valid(files_dir, db_url, DB_ROOT, 
                             mlp_db_name, train_db_name, local_root=local_root) 
                    q8 = input("\n    ??????    Continue (y/n) - (Yes by default) >")
                    if q8 == "" or q8[0] == "y" or q8[0] == "Y":
                        continue
                    else:
                        break
                
        else:
            mlp_db_name='mlp1'
            q6=input("\n    MLP hypertable name - ('%s' by default) >"%
                         mlp_db_name)
            if q6!='':
                mlp_db_name=q6

            mlp= HTdb( db_url, DB_ROOT, mlp_db_name, jwtoken= JWT_ADMIN, local_root=local_root )
            start=Table(DB_ROOT, mlp_db_name, 'Start')  
            start_rows= start.read_rows( 
                            rows="all",
                            only_fields={'BiasI','I_dim','H_dim','StoI','output_id'}
                        )

            while (True):
                print('\n     Number of starting rows    %d'%start.rows)  
                start_id= input ('\n Input Start id/ row no.  - (2 by default) - >')
                if start_id == '':
                    start_id = 2
                else:
                    start_id = int (start_id) 
                if start_id>start.rows:
                    start_id= start.rows

                start_row_num=0
                while (start_row_num==0):
                    start_row= start_rows [start_id]          

                    I_dim=          start_row['I_dim']
                    H_dim=          start_row['H_dim']
                    BiasI=          start_row['BiasI']
                    output_nrow=    start_row['output_id']

                    print('\n')  
                    print(    '             Start row    %d'%start_id)                
                    print(    '       Input nodes       I_dim       %d'%I_dim)  
                    print(    '       Hidden nodes      H_dim       %d'%H_dim) 
                    print(    '       Bias input        BiasI       %6.4f'%BiasI)
                    print(    '       Output node       output_nrow %d'%output_nrow)
                    #print('\n\n')

                    q7= input ('\n Leave this starting row (press ENTER) or choose another one (enter a new number) - >')

                    if q7 == '':
                        start_row_num= start_id
                    else:
                        start_id = int (q7) 
                        if start_id>start.rows:
                            start_id= start.rows


                mlp.close()

                two_layer_p=mlp_load_exec.mlp_load_RAM(
                    db_url, DB_ROOT, mlp_db_name, start_row_num, timer=True, local_root=local_root)

                if two_layer_p[3]:
                    print('\nLoading MLP model in RAM - time: %10.7f'%two_layer_p[3])

                while(True):
                    try:
                        input_val= input ('\n Enter the values  for input nodes separated by comma - >')
                        i_num=input_val.split(",")
                        input_values=[int(x) for x in i_num]
                    except:
                        continue
                    else:
                        result=mlp_load_exec.mlp_execute(input_values, 
                                   weights_ItoH=two_layer_p[0], 
                                   weights_HtoO=two_layer_p[1], 
                                   BiasI=two_layer_p[2], 
                                   timer=True)
                        break
                print('\n  RESULT: %4.2f'%result[0])                                
                print('\n\n')
                if result[1]:
                    print('\nExecution time: %10.7f'%result[1])

                q8 = input("\n    ??????    Continue (y/n) - (Yes by default) >")
                if q8 == "" or q8[0] == "y" or q8[0] == "Y":
                    continue
                else:
                    break

    sys.exit(0) 
