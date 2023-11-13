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


class ModelStacker():

    def __init__(self):
        self.lr = None  ## 0 
        self.rf = None  ## 1 
        self.nn = None  ## 2, this time, the architecture will be adopted as AlexNet
        self.datasets = None
        self.in_shape = None
        self.cls_nums = None


        self.BATCH_SIZE = 64
    
    def __split_into_2(self, x_data, y_data):
        n = len(x_data)
        split_point = n // 2
        return [(x_data[:split_point], y_data[:split_point]),(x_data[split_point:], y_data[split_point:])]


    def load_dataset(self, x_data, y_data):
        ## split the dataset into 3 disjoint sets
        ## suppose the type of dataset is np.array

        self.in_shape = x_data.shape[-1]
        self.cls_nums = max(y_data) + 1 

        n = len(x_data)
        n_3 = n // 3
        self.datasets = [ 
            self.__split_into_2(x_data[:n_3], y_data[:n_3]),
            self.__split_into_2(x_data[n_3:2*n_3], y_data[n_3:2*n_3]) ,
            self.__split_into_2(x_data[2*n_3:], y_data[2*n_3:])
        ]

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


    def train_models(self):
        pass




    def train_lr(self, save_path):
        train_data, test_data = self.datasets[0]
        train_loader,test_loader,(in_shape, cls_nums) = self._make_all_dataloaders_for_lr(train_data, test_data)
        self.lr = LogisticRegression( in_shape, cls_nums)
        train_model(train_loader, test_loader, self.lr, lr=0.01 , epoch=2000)
        torch.save(self.lr, save_path )
        print("model lr saved.")


    def train_rf(self, save_path):
        self.rf = RandomForestClassifier()
        # data_x, data_y = self.datasets[1]
        train_x, train_y  = self.datasets[1][0]
        test_x, test_y = self.datasets[1][1]
        self.rf.fit(train_x, train_y)
        with open(save_path, "wb") as f:
            pickle.dump(self.rf, f)

        print("test acc:"+str(self.rf.score(train_x, train_y)))
        print("train acc:"+str(self.rf.score(test_x, test_y)))
        print("model rf saved!")


    def train_nn(self, save_path, use_FCNN=False):
        if use_FCNN:
            self.nn = FCNN( self.in_shape, self.cls_nums)
        else:
            self.nn = basic_NN( self.in_shape, self.cls_nums)
        train_x_data, train_y_data = self.datasets[2][0]
        test_x_data, test_y_data = self.datasets[2][1]
        train_dataset = wrap_as_pytorch_dataset(train_x_data, train_y_data)
        test_dataset = wrap_as_pytorch_dataset(test_x_data, test_y_data)
        # train_data, test_data = split_dataset_pytorch(dataset,2)
        train_loader = wrap_as_pytorch_loader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = wrap_as_pytorch_loader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=False )
        train_model(train_loader, test_loader, self.nn, lr=0.001 , epoch=1000)
        torch.save(self.nn, save_path)
        print("model nn saved.")

    def predict(self):
        pass
        # def stack_model(models, x):
        #     ### TODO
        #     ### cast the input type accordingly
        #     ### suppose that x is a torch tensor
        #     x1 = x
        #     x2 = x.numpy()
        #     ### layer 1 
        #     with torch.no_grad():
        #         pred1 = models[0](x1)  ## NN
        #     print(pred1)
        #     pred2 = models[1].predict(x2)  ## Random Forest  
        #     print(pred2)
        #     ### combined pred1 and pred2
        #     pred = torch.cat( (pred1, torch.Tensor(pred2)) , 1 )
        #     print(pred.shape)
        #     ### layer 2 
        #     with torch.no_grad():
        #         pred  = models[2](pred)       ## LR
        #     return pred

    def collect_stacking_model_outputs(self, train_data, test_data, train_output_save_path, test_output_save_path):
        train_loader,test_loader,(in_shape, cls_nums) = self._make_all_dataloaders_for_lr(train_data, test_data)

        member_socres = collect_model_outputs(train_loader, self.lr, CUDA=True)
        non_member_socres = collect_model_outputs(test_loader, self.lr, CUDA=True)
        pickle.dump(member_socres, open(train_output_save_path,"wb"))
        pickle.dump(non_member_socres, open(test_output_save_path,"wb"))



    def _make_dataset_for_lr(self, data):
        x_data, y_data = data
        print("[1]."+str(x_data.shape))
        print("[2]."+str(y_data.shape))
        dataset = wrap_as_pytorch_dataset(x_data, y_data)
        loader = wrap_as_pytorch_loader(dataset=dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        y_data_from_rf = self.rf.predict_proba(x_data)
        y_data_from_nn = collect_model_outputs(loader, self.nn, CUDA=True)
        print("[3]."+str(y_data_from_rf.shape))
        print("[4]."+str(y_data_from_nn.shape))
        ## combine ys here
        x_data  =  np.hstack(
            (y_data_from_rf, y_data_from_nn)
        )
        print("[5]."+str(x_data.shape))

        dataset = wrap_as_pytorch_dataset(x_data, y_data)

        return dataset
        


    def _make_all_dataloaders_for_lr(self, train_data, test_data):
        
        assert self.rf is not None and self.nn is not None
        train_x_data, train_y_data = train_data
        test_x_data, test_y_data = test_data

        # ## for train x
        # dataset = wrap_as_pytorch_dataset(train_x_data, train_y_data)
        # loader = wrap_as_pytorch_loader(dataset=dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        # y_data_from_rf = self.rf.predict_proba(train_x_data)
        # y_data_from_nn = collect_model_outputs(loader, self.nn, CUDA=True)
        # # print(y_data_from_rf.shape)
        # # print(y_data_from_nn.shape)
        # ## combine ys here
        # train_x_data  =  np.hstack(
        #     (y_data_from_rf, y_data_from_nn)
        # )

        # ## for test x
        # dataset = wrap_as_pytorch_dataset(test_x_data, test_y_data)
        # loader = wrap_as_pytorch_loader(dataset=dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        # y_data_from_rf = self.rf.predict_proba(test_x_data)
        # y_data_from_nn = collect_model_outputs(loader, self.nn, CUDA=True)
        # # print(y_data_from_rf.shape)
        # # print(y_data_from_nn.shape)
        # ## combine ys here
        # test_x_data  =  np.hstack(
        #     (y_data_from_rf, y_data_from_nn)
        # )

        # print(test_x_data.shape)

        
        ## on location
        # in_shape, cls_nums = (max(train_y_data) + 1)*2, max(train_y_data) + 1 
        train_dataset = self._make_dataset_for_lr(train_data)
        test_dataset = self._make_dataset_for_lr(test_data)
        # print("train_dataset[0].shape"+str(int(train_dataset[0][0].shape[0])))
        in_shape =  int(train_dataset[0][0].shape[0]) 
        cls_nums =  max(train_y_data) + 1 
        # test_dataset = wrap_as_pytorch_dataset(test_x_data, test_y_data)
        # print(train_x_data.dtype) 
        # print(train_y_data.dtype) 
        train_loader = wrap_as_pytorch_loader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = wrap_as_pytorch_loader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=False )

        return train_loader,test_loader,(in_shape, cls_nums)
    






def load_victim_model_dataset():
    x_data,y_data = load_location()
    # x_data_shadow, y_data_shadow = slice_dataset(x_data,y_data,0,1500) 
    x_data_shadow_1, y_data_shadow_1 = slice_dataset(x_data,y_data,0,1500) 
    x_data_shadow_2, y_data_shadow_2 = slice_dataset(x_data,y_data,3000,4500) 
    x_data_shadow = np.vstack( (x_data_shadow_1, x_data_shadow_2) )
    y_data_shadow = np.hstack( (y_data_shadow_1, y_data_shadow_2) )
    print(x_data_shadow.shape)
    print(y_data_shadow.shape)
    return x_data_shadow, y_data_shadow



if __name__ == "__main__":
    de_random(10)

    x_data, y_data = load_victim_model_dataset()
    model_stacker = ModelStacker()
    model_stacker.load_dataset(x_data, y_data)
    print(model_stacker.in_shape)
    print(model_stacker.cls_nums)
    # model_stacker.train_nn("./location_nn.pt")
    # model_stacker.train_rf("./location_rf.pkl")
    # model_stacker.load_models("","./location_rf.pkl","./location_nn.pt")
    # model_stacker.train_lr("./location_lr.pt")
    # model_stacker.load_models("./location_lr.pt","./location_rf.pkl","./location_nn.pt")

    ## 收集模型在训练集和测试集上的输出
    all_train_x_data = np.vstack((
        model_stacker.datasets[0][0][0],
        model_stacker.datasets[1][0][0],
        model_stacker.datasets[2][0][0]
    ))
    all_train_y_data = np.hstack((
        model_stacker.datasets[0][0][1],
        model_stacker.datasets[1][0][1],
        model_stacker.datasets[2][0][1]
    ))
    all_test_x_data = np.vstack((
        model_stacker.datasets[0][1][0],
        model_stacker.datasets[1][1][0],
        model_stacker.datasets[2][1][0]
    ))
    all_test_y_data = np.hstack((
        model_stacker.datasets[0][1][1],
        model_stacker.datasets[1][1][1],
        model_stacker.datasets[2][1][1]
    ))

    print(all_train_x_data.shape)
    print(all_train_y_data.shape)
    print(all_test_x_data.shape)
    print(all_test_y_data.shape)


    all_train_data = (all_train_x_data, all_train_y_data)
    all_test_data = (all_test_x_data, all_test_y_data)


    # model_stacker.collect_stacking_model_outputs(all_train_data, all_test_data)

    # # ## perform attack
    # BATCH_SIZE = 128
    # attacker = torch.load("/home/user01/exps/DAMIA/Second_stage/attackers/shadow/location_attacker_epoch_150.pt")
    # member_socres = pickle.load(open("location_victim_member_socres.pkl","rb")) 
    # non_member_socres = pickle.load(open("location_victim_non_member_socres.pkl","rb")) 
    # train_x, train_y = make_attacker_dataset(member_socres, non_member_socres)
    # train_set = wrap_as_pytorch_dataset(train_x, train_y)
    # train_loader = wrap_as_pytorch_loader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    # tests_model_acc(train_loader, attacker)



    # # model1 = NN(100,10)
    # # model2 = RabdomForest()
    # # model3 = LogisticRegression(20,10)


    # # x = np.random.randn(10,100)
    # # model2.train(x, np.array([0,1,2,3,4,5,6,7,9,8]))


    # # preds = stack_model([model1,model2,model3], torch.Tensor(x))
    # # print(preds)


    


    