# coding: utf-8
from data_utils import *
from PIL import Image
import imagehash

def get_simliarity(im1, im2):
    highfreq_factor = 1
    hash_size = 8
    img_size = hash_size * highfreq_factor
    hash1 = imagehash.phash(im1,hash_size=hash_size,highfreq_factor=highfreq_factor)
    hash2 = imagehash.phash(im2,hash_size=hash_size,highfreq_factor=highfreq_factor)  
    sim = 1 - (hash1 - hash2)/len(hash1.hash)**2
    print("similarity is %f"%(sim))
    return sim
## MNIST
mnist_train_dataloader,_ = get_target_dataloader_mnist(64)
target_mnist_train_data =  mnist_train_dataloader.dataset.dataset.data[mnist_train_dataloader.dataset.indices]

extra_mnist_train_dataloader,_ = get_extra_dataloader_mnist(64)
source_mnist_train_data =  extra_mnist_train_dataloader.dataset.dataset.data[extra_mnist_train_dataloader.dataset.indices]

mnist_target_rep = np.average(target_mnist_train_data,axis=0)
mnist_source_rep = np.average(source_mnist_train_data,axis=0)
mnist_target_rep_im = Image.fromarray(mnist_target_rep)
mnist_source_rep_im = Image.fromarray(mnist_source_rep)
get_simliarity(mnist_target_rep_im,mnist_source_rep_im)


## SVHN
svhn_train_dataloader,_ = get_target_dataloader(64)
target_svhn_train_data =  svhn_train_dataloader.dataset.dataset.data[svhn_train_dataloader.dataset.indices]

extra_svhn_train_dataloader,_ = get_extra_dataloader(64)
source_svhn_train_data =  extra_svhn_train_dataloader.dataset.dataset.data[extra_svhn_train_dataloader.dataset.indices]

# target_svhn_train_data /= 255.
# source_svhn_train_data /= 255.

target_svhn_train_data = target_svhn_train_data.transpose(0, 2, 3, 1)
source_svhn_train_data = source_svhn_train_data.transpose(0, 2, 3, 1)


svhn_target_rep = np.average(target_svhn_train_data,axis=0).astype(np.uint8)
svhn_source_rep = np.average(source_svhn_train_data,axis=0).astype(np.uint8)

# svhn_target_rep = np.average(target_svhn_train_data,axis=0).astype(np.uint8)
# svhn_source_rep = np.average(source_svhn_train_data,axis=0).astype(np.uint8)

svhn_target_rep_im = Image.fromarray(svhn_target_rep)
svhn_source_rep_im = Image.fromarray(svhn_source_rep)
svhn_target_rep_im.save("svhn_target_rep.png")                             
svhn_source_rep_im.save("svhn_source_rep.png")         
get_simliarity(svhn_target_rep_im,svhn_source_rep_im)

