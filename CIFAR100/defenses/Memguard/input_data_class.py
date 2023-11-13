import numpy as np 
import ast
import configparser
np.random.seed(100)
import os
import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage")
from data_utils import *
import scipy.io as sio
import numpy as np

def load_svhn(path):
    path = '/home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data'
    train_data = sio.loadmat(path+'/train_32x32.mat')
    test_data = sio.loadmat(path+'/test_32x32.mat')
    train_images = np.transpose(train_data["X"], (3, 0, 1, 2))
    test_images = np.transpose(test_data["X"], (3, 0, 1, 2)) 

    train_labels = train_data["y"]
    train_labels[train_labels == 10] = 0
    test_labels = test_data["y"]
    test_labels[test_labels == 10] = 0

    scalar = 1 / 255.
    train_images = train_images * scalar
    test_images = test_images * scalar


    ## for target 
    x_target_train = train_images[:20000] 
    y_target_train = train_labels[:20000]
    x_target_test = train_images[:5000] 
    y_target_test = train_labels[:5000]

    ## for shadow
    x_target_train = train_images[20000:40000] 
    y_target_train = train_labels[20000:40000]
    x_target_test = train_images[5000:10000] 
    y_target_test = train_labels[5000:10000]


    ## for extra
    x_target_train = train_images[40000:60000] 
    y_target_train = train_labels[40000:60000]
    x_target_test = train_images[10000:15000] 
    y_target_test = train_labels[10000:15000]

    return [(train_images,train_labels),(train_images,test_labels)]



## target loader
def load_victim_model_dataset():
    x_data,y_data = load()
    x_data_victim, y_data_victim = slice_dataset(x_data,y_data,0,1500) 
    n = len(x_data_victim)
    x_train,y_train = x_data_victim[:n//2],y_data_victim[:n//2]
    x_test,y_test =x_data_victim[n//2:],y_data_victim[n//2:]
    return (x_train,y_train),(x_test,y_test)


## target loader
def load_defender_model_dataset():
    x_data,y_data = load()
    x_data_victim, y_data_victim = slice_dataset(x_data,y_data,0,1500) 
    n = len(x_data_victim)
    x_train_user,y_train_user = x_data_victim[:n//2],y_data_victim[:n//2]
    x_nontrain_defender,y_nontrain_defender =x_data_victim[n//2:],y_data_victim[n//2:]

    x_train_defender=np.concatenate((x_train_user,x_nontrain_defender),axis=0)
    y_train_defender=np.concatenate((y_train_user,y_nontrain_defender),axis=0)

    label_train_defender=np.zeros([x_train_defender.shape[0]],dtype=np.int)
    label_train_defender[0:x_train_user.shape[0]]=1
    return (x_train_defender,y_train_defender,label_train_defender)

## target
def load_attacker_evaluate():
    x_data,y_data = load()
    x_evaluate_member_attacker, y_evaluate_member_attacker = slice_dataset(x_data,y_data,0,750) 
    x_evaluate_nonmember_attacker, y_evaluate_nonmumber_attacker = slice_dataset(x_data,y_data,1500,2250) 

    x_evaluate_attacker=np.concatenate((x_evaluate_member_attacker,x_evaluate_nonmember_attacker),axis=0)
    y_evaluate_attacker=np.concatenate((y_evaluate_member_attacker,y_evaluate_nonmumber_attacker),axis=0)
    
    label_evaluate_attacker=np.zeros([x_evaluate_attacker.shape[0]],dtype=np.int)
    label_evaluate_attacker[0:x_evaluate_member_attacker.shape[0]]=1
    return (x_evaluate_attacker,y_evaluate_attacker,label_evaluate_attacker)

## shadow
def input_data_attacker_adv1():

    x_data,y_data = load()
    x_train_member_attacker, y_train_member_attacker = slice_dataset(x_data,y_data,1500,2250) 
    x_train_nonmember_attacker, y_train_nonmumber_attacker = slice_dataset(x_data,y_data,2250,3000) 

    x_train_attacker=np.concatenate((x_train_member_attacker,x_train_nonmember_attacker),axis=0)
    y_train_attacker=np.concatenate((y_train_member_attacker,y_train_nonmumber_attacker),axis=0)
    label_train_attacker=np.zeros([x_train_attacker.shape[0]],dtype=np.int)
    label_train_attacker[0:x_train_member_attacker.shape[0]]=1
    return (x_train_attacker,y_train_attacker,label_train_attacker)

## shadow
def input_data_attacker_shallow_model_adv1():
    x_data,y_data = load()
    x_train_member_attacker, y_train_member_attacker = slice_dataset(x_data,y_data,1500,2550) 
    x_train_nonmember_attacker, y_train_nonmumber_attacker = slice_dataset(x_data,y_data,2550,3000) 
    # y_train_member_attacker=y_train_member_attacker-1.0
    # y_train_nonmumber_attacker=y_train_nonmumber_attacker-1.0
    return (x_train_member_attacker,y_train_member_attacker),(x_train_nonmember_attacker,y_train_nonmumber_attacker)