# HTMS_AI v.0.0
# Parameters
# © A.S.Aliev, 2022

import os
import time
import random
import copy

import numpy as np

from htms_obj      import *

from mlp_par import *


# Function 
# for loading training data, validation data and MLP 
# into RAM from HTMS database (used HTMS software - middle level API)
##
# and for :
#   - training MLP (used HTMS software - object level API),
#   - validation MLP # (used HTMS software - object level API),
#   - saving MLP into HTMS database (used HTMS software - middle and object level API)
#


def mlp_train_valid (files_dir, db_url, db_root, 
                     mlp_db_name, train_db_name, local_root=""):

    #open MLP HT 
    try:
        mlp= HT_Obj( db_url, db_root, mlp_db_name, 
                jwtoken= JWT_ADMIN, local_root=local_root )
    except:
        print('\n\n ERROR opening MLP database with path "%s"\n\n'% 
              str(db_url+'\\'+db_root+'\\'+mlp_db_name))
        sys.exit()

    start=mlp.open_table('Start')  
    input_tab=mlp.open_table('Input')
    hidden=mlp.open_table('Hidden')
    output=mlp.open_table('Output')

    start_node_id= input ('\n !!! Enter row number in start table   ("%d" by default) - >'% 1 )      
    if start_node_id == '':
        start_node_id = 1
    else:
        start_node_id=int(start_node_id)
        if start_node_id>start.rows:
            start_node_id= start.rows

    start_node_obj= Obj_RAM(start)
    start_node= start_node_obj.get_from_table(start_node_id)[0]

    I_dim=          start_node.fields['I_dim']
    H_dim=          start_node.fields['H_dim']
    LR=             start_node.fields['LR']
    epoch_count=    start_node.fields['epoch_count']
    BiasI=          start_node.fields['BiasI']
    comments=       start_node.fields['Comments']
    correctness=    start_node.fields['Correctness']
    output_node_id= start_node.fields['output_id']

    input_nodes=start_node.link_refs( link_field ='StoI') 
    input_nodes.sort(key=lambda x: x.id, reverse=False)

    print('\n\n')       
    print(    '       Input nodes       I_dim       %d'%I_dim)  
    print(    '       Hidden nodes      H_dim       %d'%H_dim) 
    if not np.isnan(LR):
        print('       Learning rate     LR          %4.2f'%LR)
    else:
        print('       Learning rate     LR          NOT DEFINED')
    if epoch_count!=None:
        print('       Epoch_count       epoch_count %d'%epoch_count)
    else:
        print('       Epoch_count       epoch_count NOT DEFINED')
    if np.isnan(BiasI):
        print('       Bias input        BiasI       NOT DEFINED')
    else:
        print('       Bias input        BiasI       %6.4f'%BiasI)
    if not np.isnan(correctness):
        print('       Correctness (calculated)      %6.3f %%'%correctness)
    else:
        print('       Correctness                   NOT CALCULATED')
    if comments:
        print('       Comments : '+comments )
    else:
        print('       Comments                      NOT DEFINED')
    print('\n\n')

    if np.isnan(LR):
        LR_new= input ('\n !!! Enter Learning rate  (1.0 - by default) - >' )      
    else:
        LR_new= input ('\n !!! Enter new Learning rate  (now - "%f") - >'% LR )      
    if LR_new != '':
        LR=float(LR_new)
    else:
        if np.isnan(LR):
            LR=1.0

    if epoch_count==None:
        epoch_count_new= input ('\n !!! Enter epoch count  (1 - by default) - >' )      
    else:
        epoch_count_new= input ('\n !!! Enter new epoch count  (now - "%d") - >'% epoch_count )      
    if epoch_count_new != '':
        epoch_count=int(epoch_count_new)
    else:
        if epoch_count==None:
            epoch_count=1

    if np.isnan(BiasI):
        BiasI_new= input ('\n !!! Enter input bias  (0.0 - by default) - >' )      
    else:
        BiasI_new= input ('\n !!! Enter new input bias  (now - "%f") - >'% BiasI )      
    if BiasI_new != '':
        BiasI=float(BiasI_new)
    else:
        if np.isnan(BiasI):
            BiasI=0.0

    sample_test=0
    sam= input ('\n !!! Number of the sample to save the state of postactivations - >')   
    if sam != '':
        sample_test = int(sam)

    print('\n\n       Learning rate     LR          %4.2f'%LR)
    print(    '       Epoch_count       epoch_count %d'%epoch_count)
    print(    '       Bias input        BiasI       %6.4f'%BiasI)
    print('\n\n')

    input_bias=1
    if BiasI== 0.0 :
        input_bias=0

    t_importingstart = time.perf_counter()   #

    try:
        train= HT_Obj( db_url, db_root, 
                  train_db_name, jwtoken= JWT_ADMIN, local_root=local_root )
    except:
        print('\n\n ERROR opening train/validation database with path "%s"\n\n'% 
              str(db_url+'\\'+db_root+'\\'+train_db_name))
        sys.exit()

    # Importing Training Data

    training=train.open_table('Training')
    training_count=training.rows

    input_cols=[training.fields[col_name][0] 
                for col_name in  training.fields 
                if col_name not in ('Back_links', 'Time_row', 'Back_weights',
                                    training.maf_name+'_output')]

    train_input={}
    for sample in range(1, training_count+1): 
        train_input[sample]=[training.r_numbers(
                                attr_num=col, 
                                num_row =sample
                                ) 
                            for col in input_cols
                            ]
        if input_bias==1:
            train_input[sample].append(BiasI)

    input_cols.sort()
    train_value={}
    output_col=training.fields[training.maf_name+'_output'][0]
    for sample in range(1, training.rows+1): 
        train_value[sample]=training.r_numbers (
                                attr_num=output_col, 
                                num_row =sample
                            )  

    # Importing validation Data
 
    validation=train.open_table('Validation')
    validation_count=validation.rows

    input_cols=[validation.fields[col_name][0]
                for col_name in validation.fields 
                if col_name not in ('Back_links', 'Time_row', 'Back_weights',
                                    validation.maf_name+'_output')]
    input_cols.sort()
    valid_input={}
    for sample in range(1, validation.rows+1): 
        valid_input[sample]=[validation.r_numbers (
                                attr_num=col, 
                                num_row =sample
                                ) 
                            for col in input_cols
                            ]
        if input_bias==1:
            valid_input[sample].append(BiasI)

    valid_value={}
    output_col=validation.fields[validation.maf_name+'_output'][0]
    for sample in range(1, validation.rows+1): 
        valid_value[sample]=validation.r_numbers (
                                    attr_num=output_col, 
                                    num_row =sample
                            )  

    train.close()

    t_importingstop = time.perf_counter()   #
    print()
    print('Importing training and validation data time:')                                    
    print(float(t_importingstop - t_importingstart)) 
    print()

    #####################
    # training
    #####################

    time3=time.time()
    t_trainingstart = time.perf_counter()   #

    # read initial weights

    weights_ItoH={}
    for i in range(1, H_dim + 1): 
        weights_ItoH[i]=[0.0 for j in range(I_dim + input_bias)]
    i_n=0
    for input_node in input_nodes:
        if input_node.fields['I_name']=='input_bias' and \
            input_bias==0:  
            continue
        i_n+=1
        hidden_nodes= input_node.weight_refs( weight_field ='ItoH')
        hidden_nodes.sort(key=lambda x: x.id, reverse=False)
        first_hidden_node_id= hidden_nodes[0].id
        h_n=0
        for hidden_node in hidden_nodes: 
            h_n+=1
            weights_ItoH[h_n][i_n-1]= hidden_node.weight

    hidden_node_obj= Obj_RAM(hidden)
    hidden_nodes_rows=(j for j in range(first_hidden_node_id,
                                        first_hidden_node_id+H_dim)  
                       )
    hidden_nodes= hidden_node_obj.get_from_table(rows=hidden_nodes_rows)
    hidden_nodes.sort(key=lambda x: x.id, reverse=False)
    #weights_HtoO={}
    weights_HtoO=[0.0 for row in range(H_dim)] 
    h_n=0
    for hidden_node in hidden_nodes:
        output_node= hidden_node.weight_refs( weight_field ='HtoO')[0]
        weights_HtoO[h_n]= output_node.weight
        h_n+=1

    #print('\n')
    #print ( weights_ItoH)        #     weights_ItoH[1 : H_dim + 1][0: I_dim + input_bias]
    #print('\n')
    #print ( weights_HtoO)        #     weights_HtoO[0 : H_dim ]

    preActiv_H_0= [0.0 for r in range (H_dim)]
    postActiv_H_0=[0.0 for r in range (H_dim)]
    for epoch in range(epoch_count):

        for sample in range(1,training_count+1):
            preActivation_H= preActiv_H_0.copy()
            postActivation_H=postActiv_H_0.copy()
            h_n=0
            for hidden_node in hidden_nodes:

                #print(train_input[sample])
                #print(weights_ItoH[h_n+1])

                preActivation_H[h_n]= np.dot(
                        train_input[sample],
                        weights_ItoH[h_n+1])
                postActivation_H[h_n] = logistic(preActivation_H[h_n])
                h_n+=1

            #print(postActivation_H)
            #print(weights_HtoO)

            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)

            #print(preActivation_O)
            #print(postActivation_O)

            target_output_sample= train_value[sample]

                #table_out.r_numbers(kerr, num_row= 1, attr_num= attr_num_SpostA_O)[sample]

            FE = postActivation_O - target_output_sample
            h_n=0

            for hidden_node in hidden_nodes:
                S_error = FE * logistic_deriv(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[h_n]
                i_n=0 
                for input_node in input_nodes:

                    if input_bias==0 and input_node.fields['I_name']=='input_bias':
                        continue
                    input_value = train_input[sample][i_n]
                    gradient_ItoH =  \
                        S_error * \
                        weights_HtoO[h_n] *  \
                        logistic_deriv(preActivation_H[h_n]) * \
                        input_value
                
                    old_weight= weights_ItoH [h_n+1][i_n]
                    new_weight= old_weight - LR * gradient_ItoH
                    weights_ItoH [h_n+1][i_n]=new_weight
                    i_n+=1

                old_weight= weights_HtoO[h_n]
                new_weight= old_weight - LR * gradient_HtoO
                weights_HtoO[h_n]= new_weight
                h_n+=1

    time4=time.time()
    t_trainingstop = time.perf_counter()      #v.2  

    #print('\n')
    #print ( weights_ItoH)

    #print('\n')
    #print ( weights_HtoO)

    print('\nTraining time:')                                    
    print(t_trainingstop - t_trainingstart)  

    #####################
    #validation
    #####################  
           
    time5=time.time()
    t_validationstart = time.perf_counter()    #v.2
    HpostA=postActiv_H_0
    OpostA=0.0
    correct_classification_count = 0
    for sample in range(1, validation_count+1):
        preActivation_H= preActiv_H_0.copy()
        postActivation_H=postActiv_H_0.copy()
        h_n=0
        for hidden_node in hidden_nodes:
            h_n+=1
            preActivation_H[h_n-1]= np.dot(
                        valid_input[sample],
                        weights_ItoH[h_n])
            postActivation_H[h_n-1] = logistic(preActivation_H[h_n-1])

        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)

        if sample== sample_test:
            #print('\n             sample %d = %s'% (sample, str(valid_input[sample]) ) )
            HpostA= postActivation_H
            OpostA= postActivation_O

        if postActivation_O > 0.5:
            output_value = 1
        else:
            output_value = 0     
        
        if output_value == valid_value[sample]:
            correct_classification_count += 1

    time6=time.time()
    t_validationstop = time.perf_counter()         

    correctness= float(correct_classification_count*100/validation_count)
    print('\nPercentage of correct classifications: %6.3f' % \
          correctness
          )
  
    print('\nValidation time:')                                
    print(t_validationstop - t_validationstart) 
    print('\n\n')

    save = input("\n ??? Save new weights in DB (y/n) - (Yes by default) >")
    if save == "" or save[0] == "y" or save[0] == "Y":

        new_start=False
        start_node_id_new= \
            input ('\n !!! Input row number in start table  (current "%d" by default) - >'% start_node_id )      
        t_savingstart = time.perf_counter() 
        if start_node_id_new == '': 
            pass
        elif int(start_node_id_new) <=start_node_id:
            start_node_id=int(start_node_id_new)
        else:
            new_start=True
            start_node_id= start.rows+1
            kerr=[]
            rc= start.row(kerr, fun='add', number=1)
            if not rc:
                print ('\n               11', kerr) 
            output_node_id=output.rows+1
            kerr=[]
            rc= output.row(kerr, fun='add', number=1)
            if not rc:
                print ('\n               12', kerr)
            last_old_input_row=input_tab.rows
            rc= input_tab.row(kerr, fun='add', number=I_dim + 1) 
            if not rc:
                print ('\n               13', kerr) 
            last_old_hidden_row=hidden.rows
            rc= hidden.row(kerr, fun='add', number=H_dim)
            if not rc:
               print ('\n               14', kerr)

        start.update_row( start_node_id, 
                          {     'I_dim':        I_dim, 
                                'H_dim':        H_dim, 
                                'LR':           LR, 
                                'epoch_count':  epoch_count, 
                                'BiasI':        BiasI,
                                'TrainingTime': float(t_trainingstop - t_trainingstart),
                                'Correctness':  correctness,
                                'output_id':    output_node_id,
                                #'Comments':     " output node # %d  " % output_node  
                          } 
        ) 

        if new_start:

            for hidden_row in range(last_old_hidden_row + 1, 
                                    last_old_hidden_row + H_dim + 1): 
                hidden.update_row(
                            hidden_row, 
                            {"H_name": "hidden_"+str(hidden_row-last_old_hidden_row),
                             #'HpostA':HpostA[hidden_row-last_old_hidden_row-1] 
                             }
                        ) 
                output_weights={ 'Output' : 
                            {output_node_id: 
                             weights_HtoO[hidden_row-last_old_hidden_row-1] 
                            } 
                    }
                hidden.update_weights( hidden_row, 'HtoO', output_weights)

            hidden_weights={'Hidden':{} }
            input_links={'Input':set() }
            for input_row in range(last_old_input_row + 1, 
                                   last_old_input_row + I_dim + 2):  #input_bias + 1):
                input_links['Input'].add( input_row )                
                if  input_row == last_old_input_row + I_dim + 1: #input_bias:
                    input_tab.update_row(
                        input_row, {"I_name": "input_bias"}) 
                    #if not input_bias:
                        #break
                else:
                    input_tab.update_row(
                        input_row, {"I_name": "input_"+str(input_row-last_old_input_row)}) 
                for hidden_row in range(last_old_hidden_row + 1, 
                                        last_old_hidden_row + H_dim + 1): 
                    if  input_row == last_old_input_row + I_dim + 1 and \
                            input_bias==0:
                                hidden_weights['Hidden'].update(
                                    {hidden_row: 0.0}
                                )
                    else:
                        hidden_weights['Hidden'].update(
                                {hidden_row: 
                                    weights_ItoH \
                                        [hidden_row-last_old_hidden_row] \
                                        [input_row-last_old_input_row-1]
                                }
                            )
                input_tab.update_weights(input_row, 'ItoH', hidden_weights)

            start.update_links( start_node_id, 'StoI', input_links)

        else:
            for hidden_node in hidden_nodes:
                output_node= hidden_node.weight_refs( weight_field ='HtoO')[0]
                hidden_node.weight_change(weight_field ='HtoO', 
                    to_RAM_objects= {output_node: weights_HtoO[hidden_node.id-1]} )
                #hidden_node.fields['HpostA']=HpostA[hidden_node.id-1], 
                for input_node in input_nodes:
                    if input_bias==0 and input_node.fields['I_name']=='input_bias':
                        continue
                    input_node.weight_change(weight_field ='ItoH', 
                        to_RAM_objects= {hidden_node: 
                                  weights_ItoH[hidden_node.id][input_node.id-1]}) 
                    input_node.save()
                hidden_node.save()

        #output_node.fields['OpostA']=OpostA
        #output_node.save()
        t_savingstop = time.perf_counter()  

        print('\nSaving time:')                                    
        print(t_savingstop - t_savingstart)  

    mlp.close()

    return


