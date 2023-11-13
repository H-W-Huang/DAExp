from torchdp import PrivacyEngine
import sys
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch import nn
import torch
import numpy as np
import os
import time
import datetime
from data_utils import *
import utils 
 
CUDA = False
TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')
BATCH_SIZE = 64


## default config for dp
delta = 1e-5
sigma = 1.0
save_model = True
max_per_sample_grad_norm = 1.0


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





def train_model(train_loader, test_loader, model, epoch=10, lr=0.001, l2=0, print_interval = 5 ):

    time_cost = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=BATCH_SIZE,
        sample_size=len(train_loader.dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
    )
    privacy_engine.attach(optimizer)

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
                x = x.cuda()
                y = y.cuda()
            # print(b_x.shape)
            output = model(x)  
            # print(loss_func)
            loss = loss_func(output, y)   # cross entropy loss
            # print("loss:"+str(loss))
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
        toc = time.time()
        time_cost += (toc - tic)
        if e % print_interval == 0:
            print("on train:")
            tests_model_acc(train_loader,model, CUDA=CUDA)
            print("on test:")
            tests_model_acc(test_loader,model, CUDA=CUDA)
            print("currently the training takes %s s"%(str(time_cost)))
        if e % 5 == 0:
            with open("dp_svhn_%d.pt"%(e),"wb") as f:
                torch.save(model, f)
            print("model is saved as "+"dp_svhn_%d.pt"%(e))
    print("the trainig takes "+str(time_cost) + "s")
    
    





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
    lr = 0.0001
    epoch = 50
    train_model(train_loader, test_loader, model, epoch, lr=lr, print_interval=print_interval)


    model = torch.load("dp_svhn_15.pt")
    tests_model_acc(train_loader,model)
    tests_model_acc(test_loader,model)
    member_socres = collect_model_outputs(train_loader, model, CUDA)
    non_member_socres = collect_model_outputs(test_loader, model, CUDA)
    pickle.dump(member_socres, open("svhn_target_member_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_target_non_member_socres.pkl","wb"))
    print("scores saved.")


if __name__ == "__main__":
    with utils.Tee("dropout_svhn_%s.log"%(TODAY)):
        run()
