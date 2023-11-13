#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
"""
import argparse
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdp import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from data_utils import *
# from model_utils import *
from basic_models import *
from data_utils import *
import time

import sys
import traceback

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()



BATCH_SIZE = 32
to_device = "cuda"
use_CUDA = False
epochs = 20
n_runs = 1
lr = 0.001

## default config for dp
delta = 1e-5
sigma = 1.0
save_model = True
max_per_sample_grad_norm = 1.0

time_cost = 0



def train( model, device, train_loader, optimizer, epoch):
    global time_cost
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
    )


def test( model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    device = torch.device(to_device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader, test_loader = get_target_dataloader_cifar(BATCH_SIZE)

    run_results = []
    for _ in range(n_runs):
        # model = basic_NN(in_shape, out_shape).to(device)
        # model = torchvision.models.alexnet(pretrained=False)
        # num_classes = 10
        # model.classifier = nn.Sequential(
        #     model.classifier[1],
        #     model.classifier[2],
        #     model.classifier[4],
        #     model.classifier[6],
        # )
        # model.classifier[-1] = nn.Linear(4096, num_classes)
        # model = model.to(device)
        
        model = AlexNet()
        model = model.to(device)
        print(model)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        privacy_engine = PrivacyEngine(
            model,
            batch_size=BATCH_SIZE,
            sample_size=len(train_loader.dataset),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )
        privacy_engine.attach(optimizer)
        for epoch in range(1, epochs + 1):
            tic = time.time()
            train( model, device, train_loader, optimizer, epoch)
            toc = time.time()
            if epoch == 1 or epoch % 5 == 0 :
                test( model, device, test_loader) 
                with open("dp_cifar100_%d.pt"%(epoch),"wb") as f:
                    torch.save(model, f)
                print("model is saved as "+"dp_cifar100_%d.pt"%(epoch))
                print("currently the training takes %s s"%(str(toc - tic)))

        run_results.append(test( model, device, test_loader))

    if len(run_results) > 1:
        print("Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
            len(run_results),
            np.mean(run_results) * 100,
            np.std(run_results) * 100
        )
        )

    repro_str = (
        f"resnet50_cifar100_{lr}_{sigma}_"
        f"{max_per_sample_grad_norm}_{BATCH_SIZE}_{epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if save_model:
        torch.save(model, "cifar100_dp.pt")
    tests(model, device, train_loader)
    tests(model, device, test_loader)


if __name__ == "__main__":
    TODAY = datetime.date.today().strftime('%Y_%m_%d_%H')
    with Tee("dp_cifar100_%s.log"%(TODAY)):
        main()
    
    # use_CUDA = True
    # BATCH_SIZE = 16
    # train_loader, test_loader = get_target_dataloader(BATCH_SIZE)
    # model = torch.load("dp_cifar100_20.pt")
    # member_socres = collect_model_outputs(train_loader, model, CUDA=use_CUDA)
    # non_member_socres = collect_model_outputs(test_loader, model, CUDA=use_CUDA)
    
    # import pickle
    # pickle.dump(member_socres, open("cifar100_target_member_socres.pkl","wb"))
    # pickle.dump(non_member_socres, open("cifar100_target_non_member_socres.pkl","wb"))
    
