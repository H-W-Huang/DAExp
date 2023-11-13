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

def test(model, target_test_loader):
    model.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc




def run():

    batch_size = 64
    source_train_loader, source_test_loader = get_extra_dataloader(batch_size)
    target_train_loader, target_test_loader = get_target_dataloader(batch_size)
    with open("svhn_damia_3.pt","rb") as f:
        model = torch.load(f)

    train_acc_source = test(model, source_train_loader)
    test_acc_source = test(model, source_test_loader)
    train_acc_target = test(model, target_train_loader)
    test_acc_target = test(model, target_test_loader)
    print("train acc source:"+str(train_acc_source))
    print("test acc source:"+str(test_acc_source))
    print("train acc target:"+str(train_acc_target))
    print("test acc target:"+str(test_acc_target))

    member_socres = collect_model_outputs_DAMIA(target_train_loader, model, CUDA=True)
    non_member_socres = collect_model_outputs_DAMIA(target_test_loader, model, CUDA=True)
    pickle.dump(member_socres, open("svhn_target_member_socres_target.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_target_non_member_socres_target.pkl","wb"))

    member_socres = collect_model_outputs_DAMIA(source_train_loader, model, CUDA=True)
    non_member_socres = collect_model_outputs_DAMIA(source_test_loader, model, CUDA=True)
    pickle.dump(member_socres, open("svhn_target_member_socres_source.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_target_non_member_socres_source.pkl","wb"))

    # target_member_socres = pickle.load(open("svhn_target_member_socres.pkl","rb")) 
    # target_non_member_socres = pickle.load(open("svhn_target_non_member_socres.pkl","rb")) 

    # test_x, test_y = make_attacker_dataset(target_member_socres[:5000], target_non_member_socres)
    # test_set = wrap_as_pytorch_dataset(test_x, test_y)
    # test_loader = wrap_as_pytorch_loader(dataset=test_set, batch_size=batch_size, shuffle=False)
    # tests_model_acc(test_loader, attacker)

    #Building SVHN data loader with 1 workers
    # Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/train_32x32.mat
    # Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/test_32x32.mat
    # Building SVHN data loader with 1 workers
    # Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/train_32x32.mat
    # Using downloaded and verified file: /home/user01/exps/DAMIA/Third_stage/SVHN/dataset/svhn-data/test_32x32.mat


    mia_attacker = MIAAttacker()
    mia_attacker.load_attack_model("/home/user01/exps/DAMIA/Third_stage/SVHN/attacker/svhn_attacker_epoch_100.pt")
    mia_attacker.perfrom_attack("svhn_target_member_socres_target.pkl", "svhn_target_non_member_socres_target.pkl")
    mia_attacker.perfrom_attack("svhn_target_member_socres_source.pkl", "svhn_target_non_member_socres_source.pkl")






if __name__ == "__main__":
    run()

