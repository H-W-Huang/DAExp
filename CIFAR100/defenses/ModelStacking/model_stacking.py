import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
import utils 
from model_utils import *
from basic_models import *
from data_utils import *
import time


## 1. 限定所有的数据都来自 torchvision 的 cifar100
## 2. nn 可以直接使用 torch 的 loader 加载数
## 3. rf 使用或者训练前需要从数据集中取出相应的 numpy 数据再喂入
## 4. lr 的训练依赖于 nn 和 rf 的输出结果
## 5. 调用 stacked models 实际就是调用 lr，只是其输出需要经过其他模型预处理

# ---》
# 预测试，为了方便，直接接受 torch loader



class ModelStacker():

    def __init__(self):
        self.lr = None  ## 0 
        self.rf = None  ## 1 
        self.nn = None  ## 2, this time, the architecture will be adopted as AlexNet
        self.dataset_lr = None
        self.dataset_rf = None
        self.dataset_nn = None
        self.cls_nums = None
        self.BATCH_SIZE = 64

    def load_dataset_rf(self):
        train_loader, test_loader = get_outer_extra_dataloader_cifar(self.BATCH_SIZE)
        return [
            self._get_numpy_data_from_torch_loader(train_loader),
            self._get_numpy_data_from_torch_loader(test_loader)
        ]

    def load_dataset_nn(self):
        return get_extra_dataloader_cifar_2(self.BATCH_SIZE)


    def load_dataset_lr(self):
        return  get_target_dataloader_cifar(self.BATCH_SIZE)


    def load_models(self,lr_path, rf_path, nn_path):
        if os.path.exists(lr_path):
            self.lr = torch.load(lr_path)
            print("lr model loaded!")

        if os.path.exists(rf_path):
            self.rf = pickle.load(open(rf_path, "rb"))
            print("rf model loaded!")
        
        if os.path.exists(nn_path):
            self.nn = torch.load(nn_path)
            print("nn model loaded!")



    def train_lr(self, save_path):
        train_loader, test_loader = self.load_dataset_lr()
        train_loader = self._prepare_input_data_for_stacked_models(train_loader)
        test_loader = self._prepare_input_data_for_stacked_models(test_loader)
        in_shape = 200
        cls_nums = 100
        self.lr = LogisticRegression( in_shape, cls_nums)
        train_model(train_loader, test_loader, self.lr, lr=0.0001 , epoch=1000)
        torch.save(self.lr, save_path )
        print("model lr saved.")


    def train_rf(self, save_path):
        self.rf = RandomForestClassifier(n_estimators=100)
        dataset = self.load_dataset_rf()
        train_x, train_y  = dataset[0]
        test_x, test_y = dataset[1]
        print(len(train_x))
        print(len(train_y))
        print(len(train_x))
        print(len(test_y))
        tic = time.time()
        self.rf.fit(train_x, train_y)
        toc = time.time()
        print("The random forest take %s s for training."%(str(toc - tic)))
        with open(save_path, "wb") as f:
            pickle.dump(self.rf, f)

        print("train acc:"+str(self.rf.score(train_x, train_y)))
        print("test acc:"+str(self.rf.score(test_x, test_y)))
        print("model rf saved!")


    def train_nn(self, save_path, use_FCNN=False):
        if use_FCNN:
            self.nn = FCNN( self.in_shape, self.cls_nums)
        else:
            # self.nn = torchvision.models.alexnet(pretrained=False)
            # num_classes = 10
            # self.nn.classifier = nn.Sequential(
            #     self.nn.classifier[1],
            #     self.nn.classifier[2],
            #     self.nn.classifier[4],
            #     self.nn.classifier[6],
            # )
            # self.nn.classifier[-1] = nn.Linear(4096, num_classes)
            # print(self.nn)
            # self.nn = resnet34()
            self.nn = AlexNet()


        train_loader, test_loader = self.load_dataset_nn()
        train_model(train_loader, test_loader, self.nn, lr=0.0001 , epoch=100)
        torch.save(self.nn, save_path)
        print("model nn saved.")


    def _prepare_input_data_for_stacked_models(self, dataloader):
        ## rf
        x_data,y_data = self._get_numpy_data_from_torch_loader(dataloader)
        y_data_from_rf = self.rf.predict_proba(x_data)

        ## nn
        y_data_from_nn = collect_model_outputs(dataloader, self.nn, CUDA=True)
        
        print("[1]."+str(y_data_from_rf.shape))
        print("[2]."+str(y_data_from_nn.shape))
        ## combine ys here
        x_data  =  np.hstack(
            (y_data_from_rf, y_data_from_nn)
        )
        print("[3]."+str(x_data.shape))
        dataset = wrap_as_pytorch_dataset(x_data, y_data)
        dataloader = wrap_as_pytorch_loader(dataset=dataset, batch_size=self.BATCH_SIZE, shuffle=False )
        return dataloader


    def collect_stacking_model_outputs(self, train_data, test_data, train_output_save_path, test_output_save_path):
        train_loader,test_loader,(in_shape, cls_nums) = self._make_all_dataloaders_for_lr(train_data, test_data)

        member_socres = collect_model_outputs(train_loader, self.lr, CUDA=True)
        non_member_socres = collect_model_outputs(test_loader, self.lr, CUDA=True)
        pickle.dump(member_socres, open(train_output_save_path,"wb"))
        pickle.dump(non_member_socres, open(test_output_save_path,"wb"))


    def _get_numpy_data_from_torch_loader(self,dataloader):
        min_index = dataloader.dataset.indices[0]  
        size = len(dataloader.dataset.indices)
        dataset = dataloader.dataset.dataset
        x_data =  (dataset.data[min_index:min_index+size]).reshape(size, 3*32*32)
        y_data =  dataset.targets[min_index:min_index+size]
        # y_data =  dataloader.dataset.dataset.targets
        
        return (x_data,y_data)
    

    def predict(self, dataloader, dump_results=True, save_path = None, ):
        loader = self._prepare_input_data_for_stacked_models(dataloader)
        tests_model_acc(loader, self.lr)
        if dump_results:
            socres = collect_model_outputs(loader, self.lr, CUDA=True)
            pickle.dump(socres, open(save_path,"wb"))
        

if __name__ == "__main__":
    import datetime
    TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')

    de_random(0)
    model_stacker = ModelStacker()
    # with utils.Tee("modelstacking_nn_cifar100_%s.log"%(TODAY)):
    #     model_stacker.train_nn("./cifar100_nn.pt")

    # with utils.Tee("modelstacking_rf_cifar100_%s.log"%(TODAY)):
    #     model_stacker.train_rf("./cifar100_rf.pkl")

    # with utils.Tee("modelstacking_lr_cifar100_%s.log"%(TODAY)):
    #     model_stacker.load_models("","./cifar100_rf.pkl","./cifar100_nn.pt")
    #     model_stacker.train_lr("./cifar100_lr.pt")
    
    # target_train_loader, target_test_loader = get_target_dataloader_cifar(64)
    # model_stacker.load_models("./cifar100_lr.pt","./cifar100_rf.pkl","./cifar100_nn.pt")
    # model_stacker.predict(target_train_loader, dump_results=True, save_path = "cifar100_victim_member_socres.pkl")
    # model_stacker.predict(target_test_loader, dump_results=True, save_path = "cifar100_non_victim_member_socres.pkl")
    



    # # model1 = NN(100,10)
    # # model2 = RabdomForest()
    # # model3 = LogisticRegression(20,10)


    # # x = np.random.randn(10,100)
    # # model2.train(x, np.array([0,1,2,3,4,5,6,7,9,8]))


    # # preds = stack_model([model1,model2,model3], torch.Tensor(x))
    # # print(preds)


    


    