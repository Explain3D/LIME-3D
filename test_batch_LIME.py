#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:55 2021

@author: 
"""

import torch
import os
import importlib
import open3d as o3d
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from lime import lime_3d

num_drop_steps = 10
num_samples = int(10**2)
num_features = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def detach_cpu(tensor):
    return tensor.detach().numpy()

def detach_gpu(tensor):
    return tensor.detach().cpu().numpy()

def take_second(elem):
    return elem[1]

def take_first(elem):
    return elem[0]

def gen_pc_data(ori_data,segments,explain,label,filename):
    basic_path = "visu/"
    color = np.zeros([ori_data.shape[0],3])
    max_contri = 0
    min_contri = 0
    for k in explain[label]:
        if k[1] > 0 and k[1] > max_contri:
            max_contri = k[1]
        elif k[1] < 0 and k[1] < min_contri:
            min_contri = k[1]
    if max_contri > 0:
        positive_color_scale = 1/max_contri
    
    else:
        positive_color_scale = 0
    if min_contri < 0:
        negative_color_scale = 1/min_contri
    else:
        negative_color_scale = 0
    ex_sorted = sorted(explain[label],key=take_first,reverse=False)
    for i in range(segments.shape[0]):
        if ex_sorted[segments[i]][1] > 0:
            color[i][0] = ex_sorted[segments[i]][1] * positive_color_scale
        elif ex_sorted[segments[i]][1] < 0:
            color[i][2] = ex_sorted[segments[i]][1] * negative_color_scale
        else:
            color[i] = [0,0,0]
    pc_colored = np.concatenate((ori_data,color),axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_colored[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(pc_colored[:,3:6])
    o3d.io.write_point_cloud(basic_path+filename, pc)
    print("Generate point cloud", filename, "successful!") 

def reverse_points(points,segments,explain,start='positive',percentage=0.05):
    res = np.copy(points)
    if start == 'positive':
        explain.sort(key=take_second,reverse=True)
    elif start == 'negative':
        explain.sort(key=take_second,reverse=False)
    else:
        print('Wrong start input!')
        return res
    to_rev_list = explain
    num_of_rev = int(len(to_rev_list)*percentage)
    for i in range(num_of_rev):
        segment_to_rev = to_rev_list[i][0]
        rev_points_index = np.argwhere(segments==segment_to_rev)
        for p in rev_points_index:
            res[p] = [0,0,0]
    res = np.delete(res,np.argwhere(res==[0,0,0]),0)
    return res

def reverse_points_random(points,percentage=0.05):
    res = np.copy(points)
    num_of_rev = int(points.shape[0]*percentage)
    index_to_drop = np.random.choice(np.arange(points.shape[0]),size=num_of_rev,replace=False)
    for i in range(num_of_rev):
        res[index_to_drop[i]] = [0,0,0]
    res = np.delete(res,np.argwhere(res==[0,0,0]),0)
    return res

def sampling(points, sample_size):
    points = np.unique(points, axis=0)
    num_p = points.shape[0]
    index = range(num_p)
    sampled_index = random.sample(index,sample_size)
    sampled = points[sampled_index]
    print(np.unique(sampled,axis=0).shape)
    return sampled

def get_pred(points,classifier):
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    if torch.cuda.is_available() == True:
        points = points.cuda()
        classifier = classifier.cuda()
    classifier = classifier.eval()
    pred, bf_sftmx , _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    return pred_choice, SHAPE_NAMES[pred_choice], bf_sftmx


test_file = 'data/modelnet40_normal_resampled/modelnet40_test_lime.txt'
num_class = 40
target_class = 0 #Plane
experiment_dir = 'log/classification/pointnet_cls_msg'
data_dir = 'data/modelnet40_normal_resampled'
model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 
classifier = MODEL.get_model(num_class,normal_channel=False)
if torch.cuda.is_available() == True:
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
else:
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
print(classifier.load_state_dict(checkpoint['model_state_dict']))
layer_name = list(classifier.state_dict().keys())
selected_layer = 'fc3'



total_num_ins = 0

batch_pos_recorder = []
batch_neg_recorder = []
batch_rdm_recorder = []
R2 = 0
Mean_dis = 0
Weighted_mean_dis = 0
L1_loss = 0
weighted_L1_loss = 0
L2_loss = 0
weighted_L2_loss = 0
adjusted_R2 = 0
avg_time = 0


for line in open(test_file):
    print("Processing number ", total_num_ins+1, "instances......\n")
    if total_num_ins >= 3:
        break
    ori_class = line[:line.rfind('_')]
    cur_file = data_dir + '/' + ori_class + '/' +  line[:-1] + '.txt'
    prototype =  np.expand_dims((sampling(np.loadtxt(cur_file,delimiter=',')[:,0:3], 1024)),0)
    cur_cls_num, pred_class, logits = get_pred(prototype,classifier)
    save_name =  str(total_num_ins) + '_' + ori_class + '_' + pred_class + '.ply'
    if ori_class == pred_class:    #Count sum of correctly classified instances
        total_num_ins += 1
        l = detach_gpu(cur_cls_num)[0]
        points_for_exp = np.asarray(prototype.copy().squeeze())
        if torch.cuda.is_available() == True:
            classifier = classifier.cuda()
        predict_fn = classifier.eval()
        explainer = lime_3d.LimeImageExplainer(random_state=0,num_cluster=num_features,kernel_width=0.1)
        tmp = time.time()
        explaination = explainer.explain_instance(points_for_exp, predict_fn, top_labels=1, num_features=num_features, num_samples=num_samples, random_seed=0)
        print ('Time consuming: ',time.time() - tmp,'s')
        avg_time += time.time() - tmp
        #temp, mask = explaination.get_image_and_mask(l, positive_only=False, negative_only=False, num_features=100, hide_rest=True)
        #gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,(str(start_idx)+'_'+SHAPE_NAMES[pred_val[i-start_idx]]+'_gt_is_'+SHAPE_NAMES[l]+'_'+'_lime.ply'))
        gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,'LIME/'+str(num_samples)+'_'+str(num_features)+'/correct/'+save_name)
        
        scores_pos = [detach_gpu(logits[0][l])]
        scores_neg = [detach_gpu(logits[0][l])]
        scores_rdm = [detach_gpu(logits[0][l])]
        
        for p in range(1,num_drop_steps+1):
            print("Flipping steps: ", str(p) +" of "+ str(num_drop_steps))
            res_pos = reverse_points(points_for_exp.copy(),explaination.segments,explaination.local_exp[l],start='positive',percentage=0.05*p)
            _, cur_class_name, cur_logits = get_pred(np.expand_dims(res_pos,0),classifier)
            cur_score = cur_logits[0][l]
            scores_pos.append(detach_gpu(cur_score))
            
            res_neg = reverse_points(points_for_exp.copy(),explaination.segments,explaination.local_exp[l],start='negative',percentage=0.05*p)
            _, cur_class_name, cur_logits = get_pred(np.expand_dims(res_neg,0),classifier)
            cur_score = cur_logits[0][l]
            scores_neg.append(detach_gpu(cur_score))
            
            res_rdm = reverse_points_random(points_for_exp.copy(),0.05*p)
            _, cur_class_name, cur_logits = get_pred(np.expand_dims(res_rdm,0),classifier)
            cur_score = cur_logits[0][l]
            scores_rdm.append(detach_gpu(cur_score))
        
        batch_pos_recorder.append(np.asarray(scores_pos))
        batch_neg_recorder.append(np.asarray(scores_neg))
        batch_rdm_recorder.append(np.asarray(scores_rdm))

        
        
        
            
    elif ori_class != pred_class:
        total_num_ins += 1
        l = detach_gpu(cur_cls_num)[0]
        points_for_exp = np.asarray(prototype.copy().squeeze())
        if torch.cuda.is_available() == True:
            classifier = classifier.cuda()
        predict_fn = classifier.eval()
        explainer = lime_3d.LimeImageExplainer(random_state=0,num_cluster=num_features,kernel_width=0.1)
        tmp = time.time()
        explaination = explainer.explain_instance(points_for_exp, predict_fn, top_labels=1, num_features=num_features, num_samples=num_samples, random_seed=0)
        print ('Time consuming: ',time.time() - tmp,'s')
        avg_time += time.time() - tmp
        #temp, mask = explaination.get_image_and_mask(l, positive_only=False, negative_only=False, num_features=100, hide_rest=True)
        #gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,(str(start_idx)+'_'+SHAPE_NAMES[pred_val[i-start_idx]]+'_gt_is_'+SHAPE_NAMES[l]+'_'+'_lime.ply'))
        gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,'LIME/'+str(num_samples)+'_'+str(num_features)+'/wrong/'+save_name)
        print("Local fidelity score: ", explaination.score)
    
    Mean_dis += explaination.mean_diff
    Weighted_mean_dis += explaination.weighted_mean_diff
    L1_loss += explaination.L1_loss
    weighted_L1_loss += explaination.weighted_L1_loss
    L2_loss += explaination.L2_loss
    weighted_L2_loss += explaination.weighted_L2_loss
    R2 += explaination.score
    adjusted_R2 += explaination.adjusted_R2
    
Mean_dis /= total_num_ins
Weighted_mean_dis /= total_num_ins
L1_loss /= total_num_ins
weighted_L1_loss /= total_num_ins
L2_loss /= total_num_ins
weighted_L2_loss /= total_num_ins
R2 /=  total_num_ins
adjusted_R2 /= total_num_ins
avg_time /= total_num_ins

print("Average Mean Distance: ", Mean_dis)
print("Weighted Mean Distance: ", Weighted_mean_dis)
print("Average L1 loss: ", L1_loss)
print("Average weighted L1 loss: ", weighted_L1_loss)
print("Average L2 loss: ", L2_loss)
print("Average weighted L2 loss: ", weighted_L2_loss)
print("Average R2 score: ", R2)
print("Average adjusted R2: ", adjusted_R2)
print("Average processing time: ", avg_time)


batch_pos_recorder = np.asarray(batch_pos_recorder)
batch_neg_recorder = np.asarray(batch_neg_recorder)
batch_rdm_recorder = np.asarray(batch_rdm_recorder)
tmp_nor = np.concatenate((batch_pos_recorder,batch_neg_recorder,batch_rdm_recorder),axis=1)

np.save('visu/LIME/recorder'+str(num_features)+'_'+str(num_samples)+'_'+'.npy', tmp_nor)

for i in range(batch_pos_recorder.shape[0]):
    bias = np.min(tmp_nor[i])
    batch_pos_recorder[i] -= bias
    batch_neg_recorder[i] -= bias
    batch_rdm_recorder[i] -= bias
    tmp_nor[i] -= bias
    nor = np.max(tmp_nor[i])
    batch_pos_recorder[i] /= nor
    batch_neg_recorder[i] /= nor
    batch_rdm_recorder[i] /= nor
res_mean_pos = np.mean(batch_pos_recorder,axis=0)
res_max_pos = np.max(batch_pos_recorder,axis=0)
res_min_pos = np.min(batch_pos_recorder,axis=0)

res_mean_neg = np.mean(batch_neg_recorder,axis=0)
res_max_neg = np.max(batch_neg_recorder,axis=0)
res_min_neg = np.min(batch_neg_recorder,axis=0)

res_mean_rdm = np.mean(batch_rdm_recorder,axis=0)
res_max_rdm = np.max(batch_rdm_recorder,axis=0)
res_min_rdm = np.min(batch_rdm_recorder,axis=0)

#Draw pictures
x = np.arange(0, (num_drop_steps+1)*5, 5)
flipping_figure = plt.figure(1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=0.35)

ax = plt.subplot(111, facecolor='lightgrey')


plt.ylim(0,1.1)
ax.plot(x,res_mean_pos,'-r',linewidth=1,label='Positive')
ax.plot(x,res_mean_neg,'-b',linewidth=1,label='Negative')
ax.plot(x,res_mean_rdm,'-g',linewidth=1,label='Random')
ax.grid()
plt.xlabel("Percentage of clusters reversed",fontsize=13) 
plt.ylabel("Average Scores",fontsize=13)
plt.savefig('visu/LIME_Quan/oneline_lime_'+str(num_features)+'_'+str(num_samples)+'_'+'.jpg')

ax.plot(x,res_max_pos,'-',color='#FFC0CB',linewidth=0.5)
ax.plot(x,res_min_pos,'-',color='#FFC0CB',linewidth=0.5)
ax.fill_between(x, res_min_pos, res_max_pos, facecolor='#FFC0CB', alpha=0.3)
ax.plot(x,res_max_neg,'-',color='#6495ED',linewidth=0.5)
ax.plot(x,res_min_neg,'-',color='#6495ED',linewidth=0.5)
ax.fill_between(x, res_min_neg, res_max_neg, facecolor='#6495ED', alpha=0.3)
ax.plot(x,res_max_rdm,'-',color='#32CD32',linewidth=0.5)
ax.plot(x,res_min_rdm,'-',color='#32CD32',linewidth=0.5)
ax.fill_between(x, res_min_rdm, res_max_rdm, facecolor='#32CD32', alpha=0.3)
#ax.legend(fontsize=13)
plt.savefig('visu/LIME_Quan/lime_'+str(num_features)+'_'+str(num_samples)+'_'+'.jpg')

            


