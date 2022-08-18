# HTMS_AI v.1.0
# Script for initial create HTMS database for MLP 
# © A.S.Aliev, 2022
# 

import os
import sys
import os
import time
import random
import copy

import pandas

from htms_mid_api  import *

from mlp_par import *

def create_mlp_db ( db_url, db_root, mlp_db_name='', local_root=""):

    if mlp_db_name=='':
        mlp_db_name='mlp1'
    
    # create hypertable (HT) "mlp" as HTMS tabular network data base and 
    # create all common files in directory "db_root" on remote HTMS server
    # see - https://github.com/Arselon/Cage and https://github.com/Arselon/HTMS.   

    # creation ot mlp hypertable
    mlp = HTdb( 
        server = db_url, 
        db_root = db_root,
        db_name = mlp_db_name, 
        new = True,
        jwtoken= JWT_ADMIN,
        local_root=local_root)

    # add attributes with specified data type to the "mlp" HT 

    # Multilayer perceptron data model (allows storage of the set of the
    #  different MLP variants

    # Attribute's definitions for mlp hypertable (HTMS data types see:
    #  - https://github.com/Arselon/HTMS/tree/main/htms_low and
    #  - https://github.com/Arselon/HTMS/blob/main/htms_low/htms_low_api/data_types.py

    mlp.StoI = "*link"    # links which determine the composition of the input layer 
                          # for each of the MLP variants
    mlp.I_dim = "int4"    # input layer dimensionality 
    mlp.ItoH = "*weight"  # numbered links with weights from input layer nodes to hidden nodes
    mlp.I_name= "utf50"   # symbolic name of the input node
    mlp.H_dim = "int4"    # hidden layer dimensionality 
    mlp.HtoO = "*weight"  # numbered links with weights from hidden layer nodes to output node
    mlp.H_name= "utf50"   # symbolic name of the hidden node
    mlp.output_id="int4"  # numeric name of the output node

    # tuning parameters
    mlp.BiasI =	"float4"  # value of input bias node
    mlp.epoch_count = "int4"   # epoch_count 
        #(each complete pass through the entire training set is called an "epoch")
    mlp.LR = "float4"     # learning rate - is a parameter in an training 
                          # algorithm that determines the step size at each iteration

    # results
    mlp.TrainingTime = "float4"  # common training time
    mlp.Correctness="float4"     # total correctness after validation
    mlp.Comments = "*utf"  # any comments


    # define the semantic type of link's and weight's (numbered link's) attributes 
    mlp.relation ({'StoI':'multipart', 'ItoH':'multipart',})
#----------------------------------------------------- 
  
    # create table "Start" which contans one row for each MLP variant
    class Start (Table):
        def __str__(self):
            return self.maf_name+' - '+'Start'

    # create instance "start" of table "Start" 
    start = Start(
            ht_root = db_root,
            ht_name = mlp_db_name)
    # add fields into table - from HT attributes set
    start.fields_add= {
        'I_dim',
        'StoI',
        'H_dim',
        'LR',
        'BiasI',
        'epoch_count',
        'output_id',
        'TrainingTime',
        'Correctness',
        'Comments',
        }                       
#----------------------------------------------------- 
  
    # create table "Input" which contains one row for each input node
    class Input (Table):
        def __str__(self):
            return self.maf_name+' - '+'Input'

    # create instance "input_tab" of table "Input"
    input_tab = Input(
            ht_root = db_root,
            ht_name = mlp_db_name)

    # add fields into table 
    input_tab.fields_add= {
        'I_name',
        'ItoH',
        } 
#----------------------------------------------------- 
  
    # create table "Hidden"  which contains one row for each hidden node
    class Hidden (Table):
        def __str__(self):
            return self.maf_name+' - '+'Hidden'

    # create instance "hidden" of table "Hidden"  
    hidden = Hidden(
            ht_root = db_root,
            ht_name = mlp_db_name)

    # add fields into table
    hidden.fields_add= {
        'H_name',
        'HtoO',
        } 
#----------------------------------------------------- 
  
    # create table "Output" which contains one row for each output node
    # quantity of the output nodes equals quantity of the Start table rows
    # i.e quantity of MLP variants  
    class Output (Table):
        def __str__(self):
            return self.maf_name+' - '+'Output'

    # create instance "output" of table "Output" 
    output = Output(
            ht_root = db_root,
            ht_name = mlp_db_name)
#----------------------------------------------------- 
 
    mlp.close()   
    return 

#--------------------------------------------------------------------------------

def create_train_db ( db_url, db_root, train_db_name='', local_root=""):

    if train_db_name=='':
        train_db_name='train1'
    
    # create HT "train" as HTMS tabular network data base for storage
    # of the training and validation data set

    train = HTdb( 
        server = db_url, 
        db_root = db_root,
        db_name = train_db_name, 
        new = True,
        jwtoken= JWT_ADMIN,
        local_root=local_root)

    # create table "Training"
    class Training (Table):
        def __str__(self):
            return self.maf_name+' - '+'Training'

    training = Training(
            ht_root = db_root,
            ht_name = train_db_name)
  
    # create table "Validation"
    class Validation (Table):
        def __str__(self):
            return self.maf_name+' - '+'Validaton'

    validation = Validation(
            ht_root = db_root,
            ht_name = train_db_name)
    
    train.close()    
    return

#-----------------------------------------------------------------------------
