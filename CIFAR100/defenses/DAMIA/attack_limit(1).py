import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
from data_utils import *
from backbone import *
from model_utils import *
from utils.measureUtils import AverageMeter
import utils
import torch
from MIAAttacker import MIAAttacker

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader):
    model.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            # print(correct)
            # print(len_target_dataset)
    # print(correct.cpu().numpy() / len_target_dataset )
    acc = 100. * correct.cpu().numpy() / len_target_dataset 
    return acc




def run(e):

    print("===============================%d======================================="%e)
    batch_size = 64
    source_train_loader, source_test_loader = get_extra_dataloader_cifar(batch_size, resize_224=False)
    target_train_loader, target_test_loader = get_target_dataloader_cifar(batch_size, resize_224=False)
    with open("cifar_damia_%d.pt"%e,"rb") as f:
        model = torch.load(f)

    # # print(model)
    # train_acc_source = test(model, source_train_loader)
    # test_acc_source = test(model, source_test_loader)
    # train_acc_target = test(model, target_train_loader)
    # test_acc_target = test(model, target_test_loader)
    # print("train acc source:"+str(train_acc_source))
    # print("test acc source:"+str(test_acc_source))
    # print("train acc target:"+str(train_acc_target))
    # print("test acc target:"+str(test_acc_target))


    # member_socres = collect_model_outputs_DAMIA(target_train_loader, model, CUDA=True)
    # non_member_socres = collect_model_outputs_DAMIA(target_test_loader, model, CUDA=True)
    # pickle.dump(member_socres, open("cifar100_target_member_socres_target.pkl","wb"))
    # pickle.dump(non_member_socres, open("cifar100_target_non_member_socres_target.pkl","wb"))

    # member_socres = collect_model_outputs_DAMIA(source_train_loader, model, CUDA=True)
    # non_member_socres = collect_model_outputs_DAMIA(source_test_loader, model, CUDA=True)
    # pickle.dump(member_socres, open("cifar100_target_member_socres_source.pkl","wb"))
    # pickle.dump(non_member_socres, open("cifar100_target_non_member_socres_source.pkl","wb"))


    limit = 1
    mia_attacker = MIAAttacker()
    mia_attacker.load_attack_model("/home/user01/exps/DAMIA/Third_stage/CIFAR100/attacker/cifar100_attacker_epoch_80_limit.pt")
    print(mia_attacker.model)
    print("MIA Acc on target domain:")
    mia_attacker.perfrom_attack("cifar100_target_member_socres_target.pkl", "cifar100_target_non_member_socres_target.pkl",limit=limit)
    print("MIA Acc on source domain:")
    mia_attacker.perfrom_attack("cifar100_target_member_socres_source.pkl", "cifar100_target_non_member_socres_source.pkl",limit=limit)






if __name__ == "__main__":
    # for e in range(5,95,5):
    for e in [50]:
        # print(e)
        run(e)

