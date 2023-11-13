import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
from MIAAttacker import MIAAttacker


def run():

    mia_attacker = MIAAttacker()
    mia_attacker.load_attack_model("/home/user01/exps/DAMIA/Third_stage/SVHN/attacker/svhn_attacker_epoch_300.pt")
    mia_attacker.perfrom_attack("svhn_target_member_socres.pkl", "svhn_target_non_member_socres.pkl")



if __name__ == "__main__":
    run()

