# HTMS_AI v.1.0
# Parameters
# © A.S.Aliev, 2022

import os
import time
import random
import copy

import numpy as np
import math

from htms_mid_api  import *

from mlp_par import *
from mlp import *

# Functions 
# 
# for loading MLP into RAM from HTMS database 
# (used HTMS software - middle level API)
# 
# and
#
# for executing MLP for any input data

def mlp_load_RAM(db_url, db_root, db_name, start_id=1, 
                 jwtoken= JWT_ADMIN, timer=False, local_root=""):

    if timer:
        t_load_MLP_start = time.perf_counter()  

    mlp= HTdb( db_url, db_root, db_name, 
              jwtoken= jwtoken, local_root=local_root )

    start=Table(db_root, db_name,'Start')  
    input_tab=Table(db_root, db_name,'Input')
    hidden=Table(db_root, db_name,'Hidden')
    output=Table(db_root, db_name,'Output')

    start_row_num= start_id
    if start_row_num>start.rows:
        start_row_num= start.rows

    start_row= start.read_rows(
                    from_row=start_row_num, 
                    quantity=1,  
                    only_fields={'BiasI','I_dim', 'H_dim','StoI','output_id'}
                ) [start_row_num]          

    I_dim=          start_row['I_dim']
    H_dim=          start_row['H_dim']
    BiasI=          start_row['BiasI']
    output_nrow=    start_row['output_id']
    input_links=    start_row['StoI']

    i_rows= set( li[1] for li in input_links)
    first_i_row=min(i_rows)
    input_rows=input_tab.read_rows(
                    rows=i_rows, 
                    weights_fields=True, 
                ) 

    input_bias=1
    if BiasI== 0.0 :
        input_bias=0

    # read weights

    weights_ItoH={}
    for i in range(1, H_dim + 1): 
        weights_ItoH[i]=[0.0 for j in range(I_dim + input_bias)]
    
    for i_row in range(I_dim+ input_bias):
        input_row= input_rows[first_i_row+i_row]
        #if input_bias==0 and input_row ['I_name']=='input_bias':
            #continue
        hidden_weights= list(input_row['ItoH'])
        hidden_weights.sort(key=lambda x: x[1], reverse=False)
        first_h_row=hidden_weights[0][1]
        h_row=0
        for we in hidden_weights:
            h_row+=1
            weight=we[2]
            weights_ItoH[h_row][i_row]= weight

    weights_HtoO=[0.0 for row in range(H_dim)]

    for h_row in range(H_dim):
        hidden_row=hidden.read_rows(
                        from_row=first_h_row+h_row, 
                        quantity=1, 
                        weights_fields=True, 
                   ) [first_h_row+h_row] 

        weights_HtoO[h_row]= hidden_row['HtoO'][ 0 ][2]

    pr(str(weights_ItoH))
    pr(str(weights_HtoO))
    
    mlp.close()

    if timer:
        t_load_MLP_stop = time.perf_counter()
        timer= t_load_MLP_stop-t_load_MLP_start 
        
    two_layer_p= (weights_ItoH, weights_HtoO, BiasI, timer)

    return two_layer_p

def mlp_execute(input_values, weights_ItoH, weights_HtoO, 
                BiasI, timer=False):

    I_dim=  len(weights_ItoH[1])
    H_dim=  len(weights_HtoO)


    len_input_values=len(input_values)
    if len_input_values>I_dim:
        input_values=input_values[:I_dim]
    elif len_input_values<I_dim:
        for i in range(len(input_values),I_dim):
            input_values.append(0.0)           
    if timer:
        t_executionstart = time.perf_counter()  

    preActivation_H= [0.0 for r in range (H_dim)]
    postActivation_H=[0.0 for r in range (H_dim)]

    for hidden_node_row in range(1, H_dim+1):
        preActivation_H[hidden_node_row-1]= np.dot(
                        input_values,
                        weights_ItoH[hidden_node_row])
        postActivation_H[hidden_node_row-1] = \
                logistic(preActivation_H[hidden_node_row-1])

    preActivation_O = np.dot(postActivation_H, weights_HtoO)
    postActivation_O = logistic(preActivation_O)

    output_value = postActivation_O    
    if timer:
        t_executionstop = time.perf_counter()   
        timer= t_executionstop-t_executionstart

    return  (output_value, timer)