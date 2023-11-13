import torchvision
from torchvision import datasets
from torchvision import transforms
from torch import nn
import torch
import numpy as np
import os
import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
import utils 
import time
import datetime
from data_utils import *


DEVICE=""
CUDA = True
TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')

# def get_target_dataloader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset',**kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building SVHN data loader with {} workers".format(num_workers))
#     svhn_transfrom = transforms.Compose([
#             transforms.Resize([224, 224]),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])


#     svhn_train_set = datasets.SVHN(root=data_root, split='train', download=True,transform=svhn_transfrom)
#     svhn_test_set = datasets.SVHN(root=data_root, split='test', download=True,transform=svhn_transfrom)
    
#     train_subset =  torch.utils.data.Subset(svhn_train_set, [i for i in range(0,20000)])   
#     test_subset =  torch.utils.data.Subset(svhn_test_set, [i for i in range(0,5000)])   

#     ## 取前三分之一

#     train_loader = torch.utils.data.DataLoader(
#         train_subset,
#         batch_size=batch_size, shuffle=True, **kwargs)

#     test_loader = torch.utils.data.DataLoader(
#         test_subset,
#         batch_size=batch_size, shuffle=False, **kwargs)
#     return train_loader, test_loader


def train_model(train_loader, test_loader, model, epoch=10, lr=0.001, l2=0, print_interval = 5 ):

    time_cost = 0

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)   # optimize all cnn parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # optimize all cnn parameters

    if l2 != 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    print(optimizer)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  
    print(model)
    if CUDA:
        model = model.cuda()
    for e in range(1, epoch+1):
        tic = time.time()
        model.train()
        print("epoch : %d"%e)
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            # print("Step:%d"%step)
            # y = y.type(torch.FloatTensor)
            if CUDA:
                b_x = x.cuda()
                b_y = y.cuda()
            # print(b_x.shape)
            output = model(b_x)  
            # print(loss_func)
            loss = loss_func(output, b_y)   # cross entropy loss
            # print("loss:"+str(loss))
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
        toc = time.time()
        time_cost += (toc - tic)
        if e % print_interval == 0:
            print("on train:")
            tests_model_acc(train_loader,model)
            print("on test:")
            tests_model_acc(test_loader,model)

        if e % 5 == 0:
            with open("pure_svhn_%d.pt"%(e),"wb") as f:
                torch.save(model, f)
            print("model is saved as "+"pure_svhn_%d_0227.pt"%(e))
    print("the trainig takes "+str(time_cost) + "s")
    
    


def tests_model_acc(test_loader,model,CUDA=True):
    if CUDA:
        model = model.cuda()
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for _,data in enumerate(test_loader):
            x, y = data
            if CUDA:
                x = x.cuda()
            outputs = model(x)
            outputs = outputs.cpu()
            predicted = torch.argmax(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total
    print('Accuracy: %f %%' % (100 *acc))
    return acc



def run():
    
    utils.de_random(0)

    ## prepare data
    train_loader, test_loader = get_target_dataloader(64)

    ## setup model
    model = torchvision.models.alexnet(pretrained=False)
    num_classes = 10
    model.classifier = nn.Sequential(
        model.classifier[1],
        model.classifier[2],
        model.classifier[4],
        model.classifier[6],
    )
    model.classifier[-1] = nn.Linear(4096, num_classes)
    print(model)


    ## train the model
    print_interval = 1
    epoch = 25
    lr = 0.0001
    train_model(train_loader, test_loader, model, epoch, lr=lr, print_interval=print_interval)



# if __name__ == "__main__":
#     with utils.Tee("pure_svhn_%s.log"%(TODAY)):
#         run()


##
## threshold-based attack
# Building SVHN data loader with 1 workers
# Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/train_32x32.mat
# Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/test_32x32.mat
# model accuracy for training and test- (0.99225, 0.8834)
# maximum inference accuracy is: 0.5791999999999999
# maximum inference best_precision is: 0.5482456140350878
# maximum inference best_recall is: 0.9
# maximum inference best_ratio1 is: 0.9
# maximum inference best_ratio2(as member) is: 0.7416
# maximum inference best_delta is: 0.9995531439781189