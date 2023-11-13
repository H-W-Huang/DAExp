import os
import numpy as np
import math 
import scipy
import sys  
import torch 

DEVICE = "cuda"

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