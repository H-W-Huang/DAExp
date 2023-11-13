import numpy as np
np.random.seed(1000)
import imp
import input_data_class
import keras
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
import os
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',default='location')
args = parser.parse_args()
dataset=args.dataset 
# input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')


user_label_dim=int(config[dataset]["num_classes"])
num_classes=1
save_model=True
defense_epochs=int(config[dataset]["defense_epochs"])
user_epochs=int(config[dataset]["user_epochs"])
batch_size=int(config[dataset]["defense_batch_size"])
#defense_train_testing_ratio=float(config[dataset]["defense_training_ratio"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config_gpu))

# (x_train,y_train,l_train) = input_data_class.load_defender_model_dataset_location()



####load target model
# npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_user.npz".format(user_epochs), allow_pickle=True)
# weights=npzdata['x']
# input_shape=x_train.shape[1:]
# model=fccnet.model_user(input_shape=input_shape,labels_dim=user_label_dim)
# model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
# model.set_weights(weights)

## TODO 修改输出结果源
## 1. 分别加载 pkl 文件
## 2. 合并 member 和 nonmember 作为 defender 的 x_train
########obtain confidence score on defender's training dataset
# f_train=model.predict(x_train)
import pickle
user_label_dim = 100
member_scores, y_train = pickle.load(open("score_member_user.pkl","rb")) 
non_member_scores, y_test = pickle.load(open("score_non_member_user.pkl","rb")) 
f_train = np.concatenate((member_scores[:4000],non_member_scores[:4000]),axis=0)
y_train = np.concatenate((y_train[:4000],y_test[:4000]),axis=0)
l_train = np.zeros([f_train.shape[0]],dtype=np.int)
l_train[0:4000] = 1
# x_train,y_train,l_train = 
y_train=keras.utils.to_categorical(y_train,user_label_dim)

# f_train = np.concatenate(member_scores[:4000])
# del model

#######sort the confidence score
f_train=np.sort(f_train,axis=1)
input_shape=y_train.shape[1:]

model=fccnet.model_defense(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()

b_train=f_train[:,:]
label_train=l_train[:]


# import sys
# npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_defense.npz".format(defense_epochs), allow_pickle=True)
# weights=npzdata['x']
# model.set_weights(weights)
# model.trainable=False
# print(model.evaluate(b_train, label_train, verbose=0))
# sys.exit(0)


import time
time_cost = 0
index_array=np.arange(b_train.shape[0])
batch_num=np.int(np.ceil(b_train.shape[0]/batch_size))
for i in np.arange(defense_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        b_batch=b_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,b_train.shape[0])],:]
        y_batch=label_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,label_train.shape[0])]]
        tic = time.time()
        model.train_on_batch(b_batch,y_batch)   
        toc = time.time()
        time_cost += (toc - tic)


    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        print("currently takes %d s"%(time_cost))
        scores_train = model.evaluate(b_train, label_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])    

 
#save the defense model 
# if save_model:
#     weights=model.get_weights()
#     if not os.path.exists(result_folder):
#         os.makedirs(result_folder)
#     if not os.path.exists(result_folder+"/models"):
#         os.makedirs(result_folder+"/models")
#     np.savez(result_folder+"/models/"+"epoch_{}_weights_defense.npz".format(defense_epochs),x=weights)
