import argparse
import sys
import os
sys.path.append(os.getcwd())
from core.imagedb import ImageDB
import train as train
import config as config




def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, lr_epoch_decay=[9],
                batch_size=128, use_cuda=False,load=''):

    imagedb = ImageDB(annotation_file)
    print(imagedb.num_images)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train.train_rnet(model_store_path=model_store_path, end_epoch=end_epoch, 
        imdb=gt_imdb, batch_size=batch_size, frequent=frequent, 
        base_lr=lr, lr_epoch_decay=lr_epoch_decay, use_cuda=use_cuda,load=load)

def parse_args():
    parser = argparse.ArgumentParser(description='Train  RNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(config.ANNO_STORE_DIR,config.RNET_TRAIN_IMGLIST_FILENAME), help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODEL_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('train Rnet argument:')
    print(args)

    annotation_file = "./anno_store/imglist_anno_24.txt"
    model_store_path = "./model_store"
    end_epoch = 20 # 9
    lr = 0.0005
    batch_size = 640 # 640
    lr_epoch_decay = [8]
    use_cuda = True
    frequent = 100
    load=''
    print('batchsize:',batch_size)
    print('use_cuda:',use_cuda)
    print('frequent:',frequent)
    train_net(annotation_file, model_store_path, end_epoch, frequent, lr,lr_epoch_decay, batch_size, use_cuda,load)

    # train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
    #             end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
