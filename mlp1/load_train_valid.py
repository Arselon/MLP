# HTMS_AI v.1.0
# Script for start
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

from mlp1 import create_db
from mlp_par import *

# module with function load_mlp_db which import training and
# validation data from excel tables and fill them to HTMS tables
# "training" and "validation" (see create_db module)
# and then create all tables for perceptron 

def load_mlp_db (files_dir, db_url, db_root,
                 training_data_root,
                 validation_data_root,
                 mlp_db_name,
                 train_db_name,
                 local_root=""
                 ):


        q= input ('\n Use input nodes(columns) names from excel tables with training data (y/n) - (No by default) >')
        if q=='' or q[0]=='N' or q[0]=='n':
            use_nodes_names=False
        else:
            use_nodes_names=True

#-------------------------------------------------------------------------

        def import_data(data_root, ht, table, ):

            nonlocal use_nodes_names

            excel_data = pandas.read_excel(data_root)  # nrows= I_dim
                # https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            data_columns= excel_data.columns.values.tolist()  
            num_columns =len(data_columns)
            data_list= excel_data.values.tolist()
            count =len(data_list)
            output_column= None
            for c in range(0, num_columns ):
                if data_columns[c]=='output':
                    output_column= c
                    dimension=c
                    break

            add_attrs={}
            if use_nodes_names:
                for c in range(0, dimension):
                    c_name=table.maf_name+'_'+data_columns[c]
                    add_attrs[c_name]="float4"
            else:
                add_attrs[table.maf_name+'_input__'+str(dimension)]="float4"    # generic !!!
            
            add_attrs[table.maf_name+'_output']="int4"

            attrs= ht.add_ht_attrs(add_attrs.copy())[1]

            table.update_fields(attrs.copy())

            first_data_field_num= len(table.ht.attrs)+1
            for atr in table.fields:
                if atr.find(table.maf_name)==0 and table.fields[atr][0]< first_data_field_num:
                    first_data_field_num= table.fields[atr][0]

            table.row(fun="add", number=count)

            for t_row in range(1, count+1):
                for c in range(0, dimension+1):
                    if c==dimension:
                        input_value=int(data_list[t_row-1][c])
                    else:
                        input_value=float(data_list[t_row-1][c])
                    table.w_numbers (attr_num=first_data_field_num+c, num_row= t_row, 
                        numbers=input_value
                    )
            return dimension

#-------------------------------------------------------------------------

        print('\n\n')  
        
        create_db.create_mlp_db ( db_url, db_root, 
                                 mlp_db_name=mlp_db_name, local_root=local_root)
        create_db.create_train_db ( db_url, db_root, 
                                   train_db_name=train_db_name, local_root=local_root)       

        # Importing  data  from excel table to HTMS table 

        train= HTdb( db_url, db_root, train_db_name, 
                      jwtoken= JWT_ADMIN, local_root=local_root )

        # Importing Training Data
        training=train.open_table('Training')
        I_dim=import_data(training_data_root, train, training,)
        print('\n   Dimension of the input layer  -  I_dim = %d' % I_dim) 
        training.close()
        del training

        # Importing validation Data
        validation=train.open_table('Validation')
        I_dim_valid=import_data(validation_data_root, train, validation, )
        validation.close()
        del validation

        train.close()
        del train


        #creating all tables for perceptron

        if I_dim!=I_dim_valid:
            print('\n  ERROR!!!  Quantity of the input nodes for training and validation are different.') 
            print('\n\n') 
            return False
        
        H_dim= input ('\n Enter dimension of the hidden layer  ("%d" recommended) - >'% I_dim )      
        if H_dim == '':
            H_dim = I_dim           
        else:
            H_dim = int(H_dim)
            if H_dim >= I_dim*2:
                H_dim = I_dim*2-1

        print('\n   Dimension of the hidden layer  -  H_dim = %d' % H_dim) 

        mlp= HTdb( db_url, db_root, mlp_db_name, 
                    jwtoken= JWT_ADMIN,local_root= local_root)

        start=mlp.open_table('Start')  
        input_tab=mlp.open_table('Input')
        hidden=mlp.open_table('Hidden')
        output=mlp.open_table('Output')

        start_node_id=1
        kerr=[]
        rc= start.row(kerr, fun='add', number=1)
        if not rc:
            print ('\n               1', kerr) 

        output_node_id=1
        kerr=[]
        rc= output.row(kerr, fun='add', number=1)
        if not rc:
            print ('\n               2', kerr) 

        kerr=[]
        rc= input_tab.row(kerr, fun='add', number=I_dim + 1)
        if not rc:
            print ('\n               3', kerr) 

        kerr=[]
        rc= hidden.row(kerr, fun='add', number=H_dim + 1)
        if not rc:
            print ('\n               4', kerr) 
        """
        add_attrs={'HpostA':"float4"}
        attrs= mlp.add_ht_attrs(add_attrs)[1]
        hidden.update_fields(attrs.copy())

        add_attrs={'OpostA':"float4"}
        attrs= mlp.add_ht_attrs(add_attrs)[1]
        output.update_fields(attrs.copy())
        """
        start.update_row ( start_node_id, {
                                'I_dim':        I_dim, 
                                'H_dim':        H_dim, 
                                'output_id':    output_node_id,
                                #'Comments':     ""
                              } )        

        time1=time.time()

        # generating two weight matrices (ItoH and HtoO) with random values
        # random.uniform(-1, 1)
    
        input_links={'Input':set() }
        input_bias=1
        for input_row in range(1, I_dim + input_bias + 1):   
            if input_row == I_dim + input_bias:
                input_tab.update_row(
                    input_row, {"I_name": "input_bias"}) 
            else:
                input_tab.update_row(
                    input_row, {"I_name": "input_"+str(input_row)}) 

            update_weights={}
            for hidden_row in range (1, H_dim + 1):   
                update_weights.update({ hidden_row: random.uniform(-1, 1) })
            update_weights={ 'Hidden' : update_weights }
            input_tab.update_weights( input_row, 'ItoH', update_weights)
            #weights_StoI
            input_links['Input'].add( input_row )

        start.update_links( start_node_id, 'StoI', input_links)

        for hidden_row in range(1, H_dim + 1):   
            hidden.update_row(
                    hidden_row, {"H_name": "hidden_"+str(hidden_row)}) 
            update_weights={}
            update_weights={ 'Output' : { output_node_id: random.uniform(-1, 1) } }
            hidden.update_weights( hidden_row, 'HtoO', update_weights)

        mlp.close()
        return True

