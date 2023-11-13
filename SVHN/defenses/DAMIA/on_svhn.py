import sys
import torch
import torchvision
sys.path.append("/home/user01/exps/DAMIA/Third_stage")
from data_utils import *
import models
from utils.measureUtils import AverageMeter
from utils.measureUtils import de_random
from utils.inference_utils import *
import time

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

de_random(0)


def test(model, test_loader):
    model.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_dataset = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    print('max correct: {}, accuracy{: .2f}%\n'.format(
        correct, 100. * correct / len_dataset))

def get_source_target_dataloaders():

    train_loader_source,test_loader_source = get_extra_dataloader(BATCH_SIZE)
    train_loader_target,test_loader_target = get_target_dataloader(BATCH_SIZE)


    return train_loader_source,test_loader_source,train_loader_target,test_loader_target

def save_model_outputs_of_source_and_target(victim):

    CUDA = True
    train_loader_source,test_loader_source,train_loader_target,test_loader_target = get_source_target_dataloaders()

    member_socres = collect_model_outputs_DAMIA(train_loader_source, victim, CUDA)
    non_member_socres = collect_model_outputs_DAMIA(test_loader_source, victim, CUDA)
    pickle.dump(member_socres, open("svhn_victim_damia_source_train_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_victim_damia_source_test_socres.pkl","wb"))

    member_socres = collect_model_outputs_DAMIA(train_loader_target, victim, CUDA)
    non_member_socres = collect_model_outputs_DAMIA(test_loader_target, victim, CUDA)
    pickle.dump(member_socres, open("svhn_victim_damia_target_train_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_victim_damia_target_test_socres.pkl","wb"))


def damia_on_svhn():

    time_cost = 0
    train_loader_source,test_loader_source,train_loader_target,test_loader_target = get_source_target_dataloaders()
    input_shape = train_loader_source.dataset[0][0].shape[0]

    ## create model using DA
    model = models.Transfer_Net(num_class = 10, transfer_loss='mmd', base_net="alexnet").to(DEVICE)



    ## train the model
    ### define hyper parameters as follows
    start_epoch = 1
    epoch = 50
    lr = 0.0001
    # lmda = 5  
    lmda = 2

    train_loss_clf = AverageMeter()
    train_loss_transfer = AverageMeter()
    train_loss_total = AverageMeter()

    optimizer = torch.optim.Adam([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * lr},
    ], lr= 0.001 )

    len_source_loader = len(train_loader_source)
    len_target_loader = len(train_loader_target)
    for e in range(start_epoch, start_epoch + epoch):
        tic = time.time()
        print("running epoch %d"%e)
        model.train()
        iter_source, iter_target = iter(train_loader_source), iter(train_loader_target)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + lmda * transfer_loss  ## the total loss 

            ## update weights
            loss.backward()
            optimizer.step()


            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        toc = time.time()
        time_cost += (toc - tic)

        print("acc on source train:")
        test(model, train_loader_source)
        print("acc on source test:")
        test(model, test_loader_source)
        print("acc on target train:")
        test(model, train_loader_target)
        print("acc on target test:")
        test(model, test_loader_target)
        print("======================")

        # if (e+1) %  CFG['save_interval'] == 0 :
        #     save_model(model, optimizer, e+1, save_path)
    # torch.save(model,"DAMIA_svhn.pt")
    print("time cost:"+str(time_cost))


damia_on_svhn()


# victim = torch.load("/home/user01/exps/DAMIA/Second_stage/DAMIA/DAMIA_svhn.pt")
# train_loader_source,test_loader_source,train_loader_target,test_loader_target = get_source_target_dataloaders()
# # test(victim, train_loader_source)
# # test(victim, test_loader_source)
# # test(victim, train_loader_target)
# # test(victim, test_loader_target)
# # save_model_outputs_of_source_and_target(victim)
# train_acc, test_acc, inference = do_inference_via_confidence(victim, train_loader_source, test_loader_source)
# train_acc, test_acc, inference = do_inference_via_confidence(victim, train_loader_target, test_loader_target)
