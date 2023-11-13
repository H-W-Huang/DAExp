import torch
import torch.utils.data as Data
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pickle
import os


def load_from_pickle(p_file):
    data = None
    with open(p_file,"rb") as f:
        data = pickle.load(f)
    return data


### ========================== SVHN  start ==============================
def get_target_dataloader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset',**kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))
    svhn_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    svhn_train_set = datasets.SVHN(root=data_root, split='train', download=True,transform=svhn_transfrom)
    svhn_test_set = datasets.SVHN(root=data_root, split='test', download=True,transform=svhn_transfrom)
    
    train_subset =  torch.utils.data.Subset(svhn_train_set, [i for i in range(0,20000)])   
    test_subset =  torch.utils.data.Subset(svhn_test_set, [i for i in range(0,5000)])   

    ## 取前三分之一

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_shadow_dataloader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset',**kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))
    svhn_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    svhn_train_set = datasets.SVHN(root=data_root, split='train', download=True,transform=svhn_transfrom)
    svhn_test_set = datasets.SVHN(root=data_root, split='test', download=True,transform=svhn_transfrom)
    
    ## 取三分之一至三分之二的数据
    train_subset =  torch.utils.data.Subset(svhn_train_set, [i for i in range(20000,40000)])   
    test_subset =  torch.utils.data.Subset(svhn_test_set, [i for i in range(5000,10000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_extra_dataloader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset',**kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))
    svhn_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    svhn_train_set = datasets.SVHN(root=data_root, split='train', download=True,transform=svhn_transfrom)
    svhn_test_set = datasets.SVHN(root=data_root, split='test', download=True,transform=svhn_transfrom)
    
    train_subset =  torch.utils.data.Subset(svhn_train_set, [i for i in range(40000,60000)])   
    test_subset =  torch.utils.data.Subset(svhn_test_set, [i for i in range(10000,15000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_outer_extra_loader(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset',**kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))
    svhn_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_root='/home/user01/exps/DAMIA/Third_stage/SVHN/dataset'
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    svhn_extra_set = datasets.SVHN(root=data_root, split='extra',download=False,transform=svhn_transfrom)


    train_subset =  torch.utils.data.Subset(svhn_extra_set, [i for i in range(0,20000)])   
    test_subset =  torch.utils.data.Subset(svhn_extra_set, [i for i in range(20000,25000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)


    # x_train_set =  (svhn_extra_set.data[0:20000]).reshape(20000, 3*32*32)
    # y_train_set =  svhn_extra_set.labels[0:20000]
    # x_test_set  =  (svhn_extra_set.data[20000:25000]).reshape(5000, 3*32*32)
    # y_test_set  =  svhn_extra_set.labels[20000:25000]
    # return [
    #     (x_train_set, y_train_set),
    #     (x_test_set,  y_test_set)
    # ]
    return train_loader, test_loader
### ========================== SVHN  ends ==============================


### ========================== MNIST  start ==============================
def get_target_dataloader_mnist(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/MNIST/dataset',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    MNIST_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    mnist_train_set = datasets.MNIST(root=data_root, train=True, download=True,transform=MNIST_transfrom)
    mnist_test_set = datasets.MNIST(root=data_root, train=False, download=True,transform=MNIST_transfrom)
    
    # train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(0,8000)])   
    # test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(0,2000)])   
    # train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(0,4000)])   
    # test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(0,1000)])   
    train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(0,12000)])   
    test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(0,3000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_shadow_dataloader_mnist(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/MNIST/dataset',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    MNIST_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    mnist_train_set = datasets.MNIST(root=data_root, train=True, download=True,transform=MNIST_transfrom)
    mnist_test_set = datasets.MNIST(root=data_root, train=False, download=True,transform=MNIST_transfrom)


    # train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(8000,16000)])   
    # test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(2000,4000)])   
    # train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(4000,8000)])   
    # test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(1000,2000)])   

    train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(12000,24000)])   
    test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(3000,6000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_extra_dataloader_mnist(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/MNIST/dataset',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    MNIST_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    mnist_train_set = datasets.MNIST(root=data_root, train=True, download=True,transform=MNIST_transfrom)
    mnist_test_set = datasets.MNIST(root=data_root, train=False, download=True,transform=MNIST_transfrom)
    
    train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(16000,24000)])   
    test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(4000,6000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_target_dataloader_mnist(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/MNIST/dataset',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    MNIST_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    mnist_train_set = datasets.MNIST(root=data_root, train=True, download=True,transform=MNIST_transfrom)
    mnist_test_set = datasets.MNIST(root=data_root, train=False, download=True,transform=MNIST_transfrom)
    
    train_subset =  torch.utils.data.Subset(mnist_train_set, [i for i in range(24000,32000)])   
    test_subset =  torch.utils.data.Subset(mnist_test_set, [i for i in range(6000,8000)])   

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


# ======================== CIFAR100 ========================================


def get_shadow_dataloader_cifar(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/CIFAR100/dataset',resize_224=False,**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR100 data loader with {} workers".format(num_workers))
    if resize_224:
        cifar_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train_set = datasets.CIFAR100(root=data_root, train=True, download=True,transform=cifar_transfrom)
    cifar_test_set = datasets.CIFAR100(root=data_root, train=False, download=True,transform=cifar_transfrom)
    
    train_subset =  torch.utils.data.Subset(cifar_train_set, [i for i in range(16000,32000)])   
    test_subset =  torch.utils.data.Subset(cifar_test_set, [i for i in range(4000,8000)])   


    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_target_dataloader_cifar(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/CIFAR100/dataset',resize_224=False,**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR100 data loader with {} workers".format(num_workers))
    if resize_224:
        cifar_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train_set = datasets.CIFAR100(root=data_root, train=True, download=True,transform=cifar_transfrom)
    cifar_test_set = datasets.CIFAR100(root=data_root, train=False, download=True,transform=cifar_transfrom)
    
    train_subset =  torch.utils.data.Subset(cifar_train_set, [i for i in range(0,16000)])   
    test_subset =  torch.utils.data.Subset(cifar_test_set, [i for i in range(0,4000)])   


    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_extra_dataloader_cifar(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/CIFAR100/dataset', resize_224=False, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR100 data loader with {} workers".format(num_workers))

    if resize_224:
        cifar_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train_set = datasets.CIFAR100(root=data_root, train=True, download=True,transform=cifar_transfrom)
    cifar_test_set = datasets.CIFAR100(root=data_root, train=False, download=True,transform=cifar_transfrom)
    
    train_subset =  torch.utils.data.Subset(cifar_train_set, [i for i in range(32000,48000)])   
    test_subset =  torch.utils.data.Subset(cifar_test_set, [i for i in range(6000,10000)])   


    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_extra_dataloader_cifar_2(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/CIFAR100/dataset', resize_224=False, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR100 data loader with {} workers".format(num_workers))

    if resize_224:
        cifar_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train_set = datasets.CIFAR100(root=data_root, train=True, download=True,transform=cifar_transfrom)
    cifar_test_set = datasets.CIFAR100(root=data_root, train=False, download=True,transform=cifar_transfrom)
    
    train_subset =  torch.utils.data.Subset(cifar_train_set, [i for i in range(32000,42000)])   
    test_subset =  torch.utils.data.Subset(cifar_test_set, [i for i in range(6000,10000)])   


    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader



def get_outer_extra_dataloader_cifar(batch_size, data_root='/home/user01/exps/DAMIA/Third_stage/CIFAR100/dataset', resize_224=False, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR100 data loader with {} workers".format(num_workers))

    if resize_224:
        cifar_transfrom = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train_set = datasets.CIFAR100(root=data_root, train=True, download=True,transform=cifar_transfrom)
    cifar_test_set = datasets.CIFAR100(root=data_root, train=False, download=True,transform=cifar_transfrom)
    
    train_subset =  torch.utils.data.Subset(cifar_train_set, [i for i in range(38000,48000)])   
    test_subset =  torch.utils.data.Subset(cifar_test_set, [i for i in range(6000,10000)])   


    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader



def wrap_as_pytorch_loader(dataset, batch_size, shuffle=False):
    return  Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def wrap_as_pytorch_dataset(data_x, data_y):
    data_x = torch.Tensor(data_x).type(torch.FloatTensor)
    data_y = torch.Tensor(data_y).type(torch.LongTensor)
    dataset = Data.TensorDataset(data_x,data_y)
    return dataset


def slice_dataset(x_data,y_data, start_index, end_index):

    return x_data[start_index:end_index],y_data[start_index:end_index]


def split_dataset_pytorch(dataset, partition_num):
    length = len(dataset) // partition_num
    sizes = [length for i in range(partition_num-1)]
    sizes.append(len(dataset) - (partition_num-1) * length)
    datasets = torch.utils.data.random_split(dataset,sizes)
    return datasets



def collect_model_outputs(dataloader, model, CUDA= False):
    all_outputs = []
    with torch.no_grad():
        for _,data in enumerate(dataloader):
            x, y = data
            b_x = x
            if CUDA:
                b_x = x.cuda()
            outputs = model(b_x)
            outputs = torch.softmax(outputs,dim=1)
            outputs = outputs.cpu().numpy()
            # print(outputs.shape)
            all_outputs.append(outputs)
    return np.vstack(all_outputs[:])

def collect_model_outputs_DAMIA(dataloader, model, CUDA= False):
    all_outputs = []
    with torch.no_grad():
        for _,data in enumerate(dataloader):
            x, y = data
            if CUDA:
                b_x = x.cuda()
            outputs = model.predict(b_x)
            outputs = torch.softmax(outputs,dim=1)
            outputs = outputs.cpu().numpy()
            # print(outputs.shape)
            all_outputs.append(outputs)
    return np.vstack(all_outputs[:])


def make_attacker_dataset(member_scores, non_member_scores):


    x = np.vstack((member_scores,non_member_scores))
    y = np.hstack((
        np.zeros(len(member_scores),dtype=int),
        np.ones(len(non_member_scores),dtype=int)
    ))

    return x,y

