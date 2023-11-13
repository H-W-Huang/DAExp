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


# 训练 victim model （target model or user model）
cmd="python train_user_classification_model.py -dataset location"   #(替换为 exp_user_model.py)
os.system(cmd)

tic =  time.time()
## 训练 defense model, 也就是本地的成员推断模型（defender，user 自己的 attacker）
cmd="python train_defense_model_defensemodel.py  -dataset location"     # (保留)
os.system(cmd)
cmd= "python defense_framework.py -dataset location -qt evaluation "    # (保留)
os.system(cmd)
toc = time.time()
print("time cost:"+str(toc - tic)+" s")

## 训练 攻击者的 attack model
# 使用正常的数据训练得到. attacker 找了 shadow dataset, 得到 confidence socres, 用这部分 scores 训练 attack model
# 也就是说 attacker 不知道 defender 添加了 noise 
cmd="python train_attack_shadow_model.py  -dataset location -adv adv1"   #(替换为 exp_shadow_model.py)
os.system(cmd)

# ## 评估 defense 的效果
cmd=" python evaluate_nn_attack.py -dataset location -scenario full -version v0"  # (保留)
os.system(cmd)
