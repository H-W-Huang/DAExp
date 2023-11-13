
from torch import nn
import torch
import numpy as np
import os
import sys
sys.path.append("..") 
import time
import datetime
sys.path.append("../..") 
from data_utils import *
from basic_models import *
import utils 


CUDA = True
TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')




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

        if e % 10 == 0:
            mode_name = "shadow_svhn_%d.pt"%(e)
            with open(mode_name,"wb") as f:
                torch.save(model, f)
            print("model is saved as "+mode_name)
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
    # train_loader, test_loader = get_shadow_dataloader_mnist(64)
    train_loader, test_loader = get_extra_dataloader_mnist(64)

    ## setup model
    model = model = basic_NN(28*28, 10)
    print(model)


    ## train the model
    print_interval = 5
    epoch = 80
    lr = 0.0001
    train_model(train_loader, test_loader, model, epoch, lr=lr, print_interval=print_interval)



if __name__ == "__main__":
    with utils.Tee("shadow_svhn_%s.log"%(TODAY)):
        run()
