import os
import sys
sys.path.append(os.getcwd())
from core.imagedb import ImageDB
import train as train
import config as config



def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01,lr_epoch_decay=[], batch_size=128, use_cuda=True,load=''):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train.train_onet(model_store_path=model_store_path, end_epoch=end_epoch, 
        imdb=gt_imdb, batch_size=batch_size, frequent=frequent, lr_epoch_decay=lr_epoch_decay,
        base_lr=lr, use_cuda=use_cuda,load=load)

if __name__ == '__main__':

    print('train ONet argument:')

    annotation_file = "./anno_store/imglist_anno_48.txt"
    model_store_path = "./model_store"
    end_epoch = 90
    lr = 0.0005
    batch_size = 128

    use_cuda = True
    frequent = 50
    lr_epoch_decay = [8]
    load=''

    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr,lr_epoch_decay, batch_size, use_cuda,load)
