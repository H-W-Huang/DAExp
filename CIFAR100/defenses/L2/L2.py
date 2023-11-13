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
from basic_models import *
from data_utils import *

CUDA = True
TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')



def train_model(train_loader, test_loader, model, epoch=10, lr=0.001, l2=0, print_interval = 5 ):

    time_cost = 0

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)   # optimize all cnn parameters
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # optimize all cnn parameters

    # if l2 != 0:
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    momentum=0.9
    weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    # optimizer =  torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)

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
            print("currrently takes "+str( time_cost)+"s")

        if e % 5 == 0:
            with open("l2_cifar100_%d_alex.pt"%(e),"wb") as f:
                torch.save(model, f)
            print("model is saved as "+"l2_cifar100_%d_alex.pt"%(e))
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
    train_loader, test_loader = get_target_dataloader_cifar(64)

    ## setup model
    # model = torchvision.models.alexnet(pretrained=False)
    # num_classes = 100
    # model.classifier = nn.Sequential(
    #     model.classifier[1],
    #     model.classifier[2],
    #     model.classifier[4],
    #     model.classifier[6],
    # )
    # model.classifier[-1] = nn.Linear(4096, num_classes)
    model = AlexNet()
    print(model)
    

    # train the model
    print_interval = 5
    epoch = 100
    lr = 0.0001
    train_model(train_loader, test_loader, model, epoch, lr=lr, l2=0.001, print_interval=print_interval)


    model = torch.load("l2_cifar100_100_alex.pt")
    tests_model_acc(train_loader,model)
    tests_model_acc(test_loader,model)
    member_socres = collect_model_outputs(train_loader, model, CUDA)
    non_member_socres = collect_model_outputs(test_loader, model, CUDA)
    pickle.dump(member_socres, open("cifar100_target_member_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("cifar100_target_non_member_socres.pkl","wb"))
    print("scores saved.")



if __name__ == "__main__":
    with utils.Tee("l2_cifar_%s_alex.log"%(TODAY)):
        run()
