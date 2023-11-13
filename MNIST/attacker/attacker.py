import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
from data_utils import *
from backbone import *
from model_utils import *
import utils


def train_attacker(batch_size, save_model = True):

    ## train_data

    epoch = 150
    shadow_member_socres = pickle.load(open("svhn_shadow_member_socres.pkl","rb")) 
    shadow_non_member_socres = pickle.load(open("svhn_shadow_non_member_socres.pkl","rb")) 

    target_member_socres = pickle.load(open("svhn_target_member_socres.pkl","rb")) 
    target_non_member_socres = pickle.load(open("svhn_target_non_member_socres.pkl","rb")) 

    # train_x, train_y = make_attacker_dataset(shadow_member_socres[:2000], shadow_non_member_socres)
    train_x, train_y = make_attacker_dataset(shadow_member_socres[:3000], shadow_non_member_socres)
    train_set = wrap_as_pytorch_dataset(train_x, train_y)
    train_loader = wrap_as_pytorch_loader(dataset=train_set, batch_size=batch_size, shuffle=True)

    # test_x, test_y = make_attacker_dataset(target_member_socres[:2000], target_non_member_socres)
    test_x, test_y = make_attacker_dataset(target_member_socres[:3000], target_non_member_socres)
    test_set = wrap_as_pytorch_dataset(test_x, test_y)
    test_loader = wrap_as_pytorch_loader(dataset=test_set, batch_size=batch_size, shuffle=False)

    attacker = FCNN(10,2)
    train_model(train_loader, test_loader, attacker, lr=0.00001 , epoch=epoch)

    if save_model:
        torch.save(attacker,"svhn_attacker_epoch_%d.pt"%(epoch))




def run():
    utils.de_random(0)
    train_loader, test_loader = get_shadow_dataloader_mnist(64)
    with open("shadow_svhn_80.pt","rb") as f:
        model = torch.load(f)
    member_socres = collect_model_outputs(train_loader, model, CUDA=True)
    non_member_socres = collect_model_outputs(test_loader, model, CUDA=True)
    pickle.dump(member_socres, open("svhn_shadow_member_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_shadow_non_member_socres.pkl","wb"))

    train_loader, test_loader = get_target_dataloader_mnist(64)
    with open("/home/user01/exps/DAMIA/Third_stage/MNIST/pure_model/pure_svhn_80.pt","rb") as f:
        model = torch.load(f)
    member_socres = collect_model_outputs(train_loader, model, CUDA=True)
    non_member_socres = collect_model_outputs(test_loader, model, CUDA=True)
    pickle.dump(member_socres, open("svhn_target_member_socres.pkl","wb"))
    pickle.dump(non_member_socres, open("svhn_target_non_member_socres.pkl","wb"))

    train_attacker(128, save_model= False)



if __name__ == "__main__":
    run()