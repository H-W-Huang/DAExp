
import sys
sys.path.append("/home/user01/exps/DAMIA/Third_stage/") 
from data_utils import *
from backbone import *
from model_utils import *
import utils
import torch

class MIAAttacker():
    
    def __init__(self):
        self.model = None
        self.batch_size = 64
    
    def load_attack_model(self, model_path):
        self.model = torch.load(model_path)
        if self.model is None:
            print("load Failed!")
    
    def perfrom_attack(self, member_socres, non_member_score):
        target_member_socres = pickle.load(open(member_socres,"rb")) 
        target_non_member_socres = pickle.load(open(non_member_score,"rb")) 
        test_x, test_y = self._make_attacker_dataset(target_member_socres[:len(target_non_member_socres)], target_non_member_socres)
        test_set = wrap_as_pytorch_dataset(test_x, test_y)
        test_loader = wrap_as_pytorch_loader(dataset=test_set, batch_size=self.batch_size , shuffle=False)
        tests_model_acc(test_loader, self.model)


    # def perform_threshold_attack(self, member_socres, non_member_score, label_member, label_non_member):
    #     target_member_socres = pickle.load(open(member_socres,"rb")) 
    #     target_non_member_socres = pickle.load(open(non_member_score,"rb")) 
    #     target_member_labels = pickle.load(open(label_member,"rb")) 
    #     target_none_member_labels = pickle.load(open(label_non_member,"rb")) 
    #     len_test = len(target_non_member_socres)
    #     target_member_socres = target_member_socres[:len_test]
    #     target_member_labels = target_member_labels[:len_test]
    #     self._inference_via_confidence(target_member_socres, target_non_member_socres, target_member_labels, target_none_member_labels)

    def perform_threshold_attack(self, model, train_loader, test_loader):
        output_test,test_label = self._get_model_output_and_label(model, test_loader)
        output_train,train_label = self._get_model_output_and_label(model, train_loader)
        max_accuracy = self._inference_via_confidence(output_train, output_test, train_label, test_label)
        return max_accuracy

    
    def _make_attacker_dataset(self, member_scores, non_member_scores):
        x = np.vstack((member_scores,non_member_scores))
        y = np.hstack((
            np.zeros(len(member_scores),dtype=int),
            np.ones(len(non_member_scores),dtype=int)
        ))
        return x,y

    def _get_model_output_and_label(self, model, data_loader):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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



    def _inference_via_confidence(self, confidence_mtx1, confidence_mtx2, label_vec1, label_vec2, threshold=-1):
    
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

    
    