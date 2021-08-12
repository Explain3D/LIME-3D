#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:27:59 2021

@author: tan
"""

import argparse
import numpy as np
import os
import torch
import logging
import sys
import importlib
from integrated_gradients import IntegratedGradients
import matplotlib.pyplot as plt
import open3d as o3d
import time

from lime import lime_3d_remove


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/shape_names.txt'))] 

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
    
def reverse_points(points,segments,explain,start='positive',percentage=0.2):
    num_input_dims = points.shape[1]
    basic_path = "output/"
    filename = 'reversed.ply'
    if start == 'positive':
        to_rev_list = np.argsort(explain)[-int(len(explain)*percentage):]
        to_rev_list = to_rev_list[::-1]
    elif start == 'negative':
        to_rev_list = np.argsort(explain)[:int(len(explain)*percentage)]
    else:
        print('Wrong start input!')
        return points
    for i in range(len(to_rev_list)):
        segment_to_rev = to_rev_list[i]
        rev_points_index = np.argwhere(segments==segment_to_rev)
        for p in rev_points_index:
            points[p] = np.zeros([num_input_dims])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:,0:3])
    o3d.io.write_point_cloud(basic_path+filename, pc)
    print("Generate point cloud", filename, "successful!") 
    return points

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    return parser.parse_args()

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

def test(model, loader, num_class=40, vote_num=1):
    if loader[-3:] == 'npy':
        points = np.load(loader)
    elif loader[-3:] == 'txt':
        points = np.loadtxt(loader,delimiter=',')
    elif loader[-3:] == 'ply':
        points = o3d.io.read_point_cloud(loader)
        points = np.asarray(points.points)
    if points.shape[1] > 3:
        points = points[:,0:3]
    if points.shape[0] > 1024:
        points = sampling(points,1024)
    points = np.expand_dims(points,0)
    points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    classifier = model.eval()
    pred, bf_sftmx , _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    print('Fc3 layer score:\n', bf_sftmx[0][pred_choice])
    print('Prediction Score:\n', pred[0])
    print('Predict Result: ',pred_choice, SHAPE_NAMES[pred_choice])
    return points, pred_choice, pred


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''MODEL LOADING'''
    num_class = 40
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class,normal_channel=args.normal)
    #checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    filename = 'data/modelnet40_normal_resampled/wardrobe/wardrobe_0002.txt'
    with torch.no_grad():
        points, pred, logits = test(classifier.eval(), filename, vote_num=args.num_votes)
        
    
    l = pred.detach().numpy()[0]
    points_for_exp = np.asarray(points.squeeze().transpose(1,0))
    predict_fn = classifier.eval()
    explainer = lime_3d_remove.LimeImageExplainer(random_state=0)
    tmp = time.time()
    explaination = explainer.explain_instance(points_for_exp, predict_fn, top_labels=5, num_features=20, num_samples=10, random_seed=0)
    print ('Time consuming: ',time.time() - tmp,'s')
    #temp, mask = explaination.get_image_and_mask(l, positive_only=False, negative_only=False, num_features=100, hide_rest=True)
    #gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,(str(start_idx)+'_'+SHAPE_NAMES[pred_val[i-start_idx]]+'_gt_is_'+SHAPE_NAMES[l]+'_'+'_lime.ply'))
    gen_pc_data(points_for_exp,explaination.segments,explaination.local_exp,l,'test_lime.ply')
    #return mask,points
    return explaination
    
    

    
if __name__ == '__main__':
    args = parse_args()
    exp = main(args)

    