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

CUDA = True
TODAY = datetime.date.today().strftime('%Y_%m_%d_%H_%M')

def get_target_dataloader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/MNIST/dataset',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    MNIST_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    mnist_train_set = datasets.MNIST(root=data_root, train=True, download=True,transform=MNIST_transfrom)
    mnist_test_set = datasets.MNIST(root=data_root, train=False, download=True,transform=MNIST_transfrom)
    
    train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(0,8000)])   
    test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(0,2000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


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
            print("model is saved as "+"pure_svhn_%d.pt"%(e))
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


import torch.nn.functional as F
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"

def run():
    
    utils.de_random(0)

    ## prepare data
    train_loader, test_loader = get_target_dataloader(64)

    ## setup model
    model = basic_NN(28*28, 10)
    print(model)


    ## train the model
    print_interval = 5
    epoch = 800
    lr = 0.001
    train_model(train_loader, test_loader, model, epoch, lr=lr, print_interval=print_interval)



# if __name__ == "__main__":
#     with utils.Tee("pure_mnist_%s.log"%(TODAY)):
#         run()



def inference_via_confidence(confidence_mtx1, confidence_mtx2, label_vec1, label_vec2, threshold=-1):
    
    #----------------First step: obtain confidence lists for both training dataset and test dataset--------------
    confidence1 = []
    confidence2 = []
    acc1 = 0
    acc2 = 0
    for num in range(confidence_mtx1.shape[0]):
        confidence1.append(confidence_mtx1[num,label_vec1[num]])  ## 取出正确标签在输出结果中的概率值
        if np.argmax(confidence_mtx1[num,:]) == label_vec1[num]:
            acc1 += 1
            
    for num in range(confidence_mtx2.shape[0]):
        confidence2.append(confidence_mtx2[num,label_vec2[num]])
        if np.argmax(confidence_mtx2[num,:]) == label_vec2[num]:
            acc2 += 1
    confidence1 = np.array(confidence1)
    confidence2 = np.array(confidence2)
    
    print('model accuracy for training and test-', (acc1/confidence_mtx1.shape[0], acc2/confidence_mtx2.shape[0]) )
    
    
    #sort_confidence = np.sort(confidence1)
    sort_confidence = np.sort(np.concatenate((confidence1, confidence2)))
    max_accuracy = 0.5
    best_precision = 0.5
    best_recall = 0.5
    best_ratio1 = 0
    best_ratio2 = 0
    best_delta = None

    
    # delta = 0.9998681545257568  ## threshold
    # ratio1 = np.sum(confidence1>=delta)/confidence_mtx1.shape[0]  ## training sampler as member
    # ratio2 = np.sum(confidence2>=delta)/confidence_mtx2.shape[0]  ## test sample as member
    # accuracy_now = 0.5 * ( ratio1 + (1-ratio2) )
    # max_accuracy = accuracy_now
    # best_precision = ratio1/(ratio1+ratio2)
    # best_recall = ratio1
    # best_ratio1 = ratio1
    # best_ratio2 = ratio2
    # best_delta = delta
    if threshold == -1 :
        for num in range(len(sort_confidence)):
            delta = sort_confidence[num]  ## threshold
            ratio1 = np.sum(confidence1>=delta)/confidence_mtx1.shape[0]  ## training sampler as member
            ratio2 = np.sum(confidence2>=delta)/confidence_mtx2.shape[0]  ## test sample as member
            accuracy_now = 0.5 * ( ratio1 + (1-ratio2) )
            # accuracy_now = 0.5 * ( ratio1 + (1-ratio2) )
            if accuracy_now > max_accuracy:
                max_accuracy = accuracy_now
                best_precision = ratio1/(ratio1+ratio2)
                best_recall = ratio1
                best_ratio1 = ratio1
                best_ratio2 = ratio2
                best_delta = delta
    else:
        delta = threshold  ## threshold
        ratio1 = np.sum(confidence1>=delta)/confidence_mtx1.shape[0]  ## training sampler as member
        ratio2 = np.sum(confidence2>=delta)/confidence_mtx2.shape[0]  ## test sample as member
        accuracy_now = 0.5 * ( ratio1 + (1-ratio2) )
        max_accuracy = accuracy_now
        best_precision = ratio1/(ratio1+ratio2)
        best_recall = ratio1
        best_ratio1 = ratio1
        best_ratio2 = ratio2
        best_delta = delta
    print('maximum inference accuracy is:', max_accuracy)
    print('maximum inference best_precision is:', best_precision)
    print('maximum inference best_recall is:', best_recall)
    print('maximum inference best_ratio1 is:', best_ratio1)
    print('maximum inference best_ratio2(as member) is:', best_ratio2)
    print('maximum inference best_delta is:', best_delta)
    return acc1/confidence_mtx1.shape[0], acc2/confidence_mtx2.shape[0], max_accuracy

def get_model_output_and_label(model, data_loader):
    outputs = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if hasattr(model, "predict"):
                s_output = model.predict(data)
            else:
                s_output = model(data)
            preds = torch.nn.functional.softmax(s_output, dim=1)
            outputs += preds.cpu().tolist()
            labels += target.cpu().tolist()
    outputs = np.array(outputs)
    labels = np.array(labels)
    return outputs,labels
    
def do_inference_via_confidence(model, train_loader, test_loader, threshold = -1):
    output_test,test_label = get_model_output_and_label(model, test_loader)
    output_train,train_label = get_model_output_and_label(model, train_loader)
    max_accuracy = inference_via_confidence(output_train, output_test, train_label, test_label,threshold)
    return max_accuracy

DEVICE = "cuda"
batch_size = 64
# source_train_loader, source_test_loader = get_extra_dataloader(batch_size)
train_loader, test_loader = get_target_dataloader(64)
with open("pure_svhn_95.pt","rb") as f:
    model = torch.load(f)

do_inference_via_confidence(model, train_loader, test_loader, threshold = -1)