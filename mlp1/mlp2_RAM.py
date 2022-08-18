# HTMS_AI v.0.0
# MLP v.2

# © A.S.Aliev, 2022

import os
import pandas
import numpy as np
import time

from mlp_par import *

def logistic(x):
    return 1.0/(1 + np.exp(-x))
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


def mlp_2_RAM (files_dir, ):

    LR= input ('\n Input learning rate   ("%4.2f" by default) - >'% float(0.5) )      
    if LR == '':
        LR = float(0.5)
    I_dim= input ('\n Input dimensionality of the input layer   ("%d" by default) - >'% 4 )      
    if I_dim == '':
        I_dim = 4
    H_dim= input ('\n Input dimensionality of the hidden layer   ("%d" by default) - >'% 7 )      
    if H_dim == '':
        H_dim = 7
    epoch_count= input ('\n Input epoch count   ("%d" by default) - >'% 10 )      
    if epoch_count == '':
        epoch_count = 10

    time1=time.time()
    # two weight matrices with random values
    #np.random.seed(1)
    weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
    weights_HtoO = np.random.uniform(-1, 1, H_dim)
    # empty arrays for the preactivation and postactivation values 
    preActivation_H = np.zeros(H_dim)
    postActivation_H = np.zeros(H_dim)

    training_data_root=   (os.path.join(files_dir, 'Train_MLP_Exp_2.xlsx'))
    validation_data_root= (os.path.join(files_dir, 'Valid_MLP_Exp_2.xlsx'))

    # Importing Training Data
    time2=time.time()
    
    training_data = pandas.read_excel(training_data_root)  # nrows= I_dim
        # https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
    target_output = training_data.output
    training_data = training_data.drop(['output'], axis=1)
    training_data = np.asarray(training_data)
    training_count = len(training_data[:,0])

    # Importing validation Data
    validation_data = pandas.read_excel(validation_data_root)
    validation_output = validation_data.output
    validation_data = validation_data.drop(['output'], axis=1)
    validation_data = np.asarray(validation_data)
    validation_count = len(validation_data[:,0])

    #####################
    #training
    #####################

    time3=time.time()
    t_trainingstart = time.perf_counter()          #v.2

    for epoch in range(epoch_count):
        for sample in range(training_count):
            for node in range(H_dim):
                preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
                postActivation_H[node] = logistic(preActivation_H[node])
            
            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)
        
            FE = postActivation_O - target_output[sample]
        
            for H_node in range(H_dim):
                S_error = FE * logistic_deriv(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[H_node]
                       
                for I_node in range(I_dim):
                    input_value = training_data[sample, I_node]
                    gradient_ItoH = S_error * weights_HtoO[H_node] * \
                      logistic_deriv(preActivation_H[H_node]) * input_value
                
                    weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
                
                weights_HtoO[H_node] -= LR * gradient_HtoO

    time4=time.time()
    t_trainingstop = time.perf_counter()           #v.2  

    #####################
    #validation
    #####################   
    
    time5=time.time()
    t_validationstart = time.perf_counter()        #v.2
    
    correct_classification_count = 0
    for sample in range(validation_count):
        for node in range(H_dim):
            preActivation_H[node] = np.dot(validation_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])
            
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)
        
        if postActivation_O > 0.5:
            output = 1
        else:
            output = 0     
        
        if output == validation_output[sample]:
            correct_classification_count += 1

    time6=time.time()
    t_validationstop = time.perf_counter()            #v.2

    print('\nPercentage of correct classifications: %6.3f' % \
          float(correct_classification_count*100/validation_count)
          )
    print('Training time:')                           #v.2
    print(t_trainingstop - t_trainingstart)           #v.2

    print('Validation time:')                         #v.2
    print(t_validationstop - t_validationstart)       #v.2

    print()

    return
