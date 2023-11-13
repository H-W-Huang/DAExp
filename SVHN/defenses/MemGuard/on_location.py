'''
This script is used to run the the pipeline of MemGuard. 
'''
import os 
import configparser
import time

config = configparser.ConfigParser()
config.read('config.ini')
result_folder="./result/location/code_publish/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
config["location"]["result_folder"]=result_folder
with open("config.ini",'w') as configfile:
    config.write(configfile)
    configfile.close()


## 训练 victim model （target model or user model）
cmd="python train_user_classification_model.py -dataset location"
os.system(cmd)

print("initiate MemGuard")
time_start=time.time()

# 训练 defense model, 也就是本地的成员推断模型（defender，user 自己的 attacker）
cmd="python train_defense_model_defensemodel.py -dataset location"
os.system(cmd)
cmd= "python defense_framework.py -dataset location -qt evaluation " 
os.system(cmd)
time_end=time.time()
print('time cost',time_end-time_start,'ms')


# ## 评估 defense 的效果
# cmd=" python evaluate_nn_attack.py -dataset purchase -scenario full -version v0"
# os.system(cmd)

