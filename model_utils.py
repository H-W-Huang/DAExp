import os
import torch
import torch.nn as nn
# from basic_models import *
from data_utils import *
import numpy as np 
import random
import time

# torch.manual_seed(1)    # reproducible
# Hyper Parameters

CUDA = True

def tests_model_acc(test_loader,model,CUDA=True):
    if CUDA:
        model = model.cuda()
    total = 0
    correct = 0
    # preds = []
    with torch.no_grad():
        model.eval()
        for _,data in enumerate(test_loader):
            x, y = data
            if CUDA:
                x = x.cuda()
            outputs = model(x)
            outputs = outputs.cpu()
            # preds.append(outputs.numpy())
            predicted = torch.argmax(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # print(correct)
    acc = correct / total
    print('Accuracy: %f %%' % (100 *acc))
    return acc



def train_model(train_loader, test_loader, model, epoch=10, lr=0.01, l2=0, print_interval = 5, show_best_acc = False):

    time_cost = 0
    best_acc_record = [0,0]
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
            # output = torch.softmax(output,dim=1)   ## not for BCE
            # print(output[:10])
            # print(b_y[:10])
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
            test_acc = tests_model_acc(test_loader,model)
            if best_acc_record[0] < test_acc:
                best_acc_record[0] = test_acc
                best_acc_record[1] = e
            print("currently takes %s s"%(str(time_cost)))
    print("the trainig takes "+str(time_cost) + "s")
    print("the best acc is %f in epoch %d"%(best_acc_record[0], best_acc_record[1]))






def de_random(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




# if __name__ == "__main__":
#     de_random(1)
#     partition_num = 2 
#     BATCH_SIZE = 128

#     data_x, data_y = load_location()
#     # data_x, data_y = slice_dataset(data_x,data_y,0,10000)
#     # data_x, data_y = load_purchase()
#     # data_x, data_y = load_texas()
#     print(data_x.shape)
#     dataset = wrap_as_pytorch_dataset(data_x, data_y)
#     train_data, test_data = split_dataset_pytorch(dataset,2)
#     train_loader = wrap_as_pytorch_loader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = wrap_as_pytorch_loader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

#     net = basic_NN(data_x.shape[1], max(data_y)+1)
#     train_model(train_loader, test_loader, net, lr=0.001 , epoch=5)