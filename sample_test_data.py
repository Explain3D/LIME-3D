# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:13:16 2021

@author: 
"""

import numpy as np
import random
import os

test_file = 'modelnet40_test.txt'
SHAPE_NAMES = [line.rstrip() for line in \
    open('../shape_names.txt')] 
num_cls = 40
num_adv_data_to_test = 1200

num_test_in_cls = num_adv_data_to_test//num_cls
data = []
total_test_list = []
with open(test_file,'r') as f:
    for line in f:
        data.append(list(line.strip('\n').split(',')))
        
total_test_list = data



if os.path.exists('modelnet40_test_lime.txt') == False:
    data_index = random.sample(range(0,len(total_test_list)), 1000)
    data_index = np.unique(data_index)
    
# =============================================================================
#     for cn in SHAPE_NAMES:
#         cur_data_list = []
#         for ins in data:
#             if ins[0][:-5] == cn:
#                 cur_data_list.append(ins)
#         print(cur_data_list)
#         if len(cur_data_list) > num_test_in_cls:
#             adv_data = random.sample(range(0,len(cur_data_list)), num_test_in_cls)
#         else:
#             adv_data = range(0,len(cur_data_list))
#         adv_data = np.unique(adv_data)
# =============================================================================
    for i in range(len(data_index)):
        cur_lime_data = total_test_list[data_index[i]]
        f = open('modelnet40_test_lime.txt', 'a')
        f.write(str(cur_lime_data[0][:]) + '\n')
        f.close()
                
        
    
else:
    print('File already exists!')
