import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from model import MIDL
from train import train
from data_set import MyDataset
from utils.dataset import BaseDataset
from torch.utils.data import DataLoader

import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
from utils.data_utils import PadCollate_without_know

import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained("/mnt/afs/real_cyz/zzc/clip-vit-base-patch32")

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MIDL', type=str, help='the model name', choices=['MV_CLIP','MIDL'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='./output_dir/', type=str, help='the output path')
    parser.add_argument('--name', default=None, type=str, help='model name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)




    if args.model == 'MIDL':
        use_np = False
        annotation_files = './text_data'
        img_files = "./img_emb"
        annotation_train = os.path.join(annotation_files, "mmsd_traindep.json")
        annotation_val = os.path.join(annotation_files, "mmsd_validdep.json")
        annotation_test = os.path.join(annotation_files, "mmsd_testdep.json")
        img_train = os.path.join(img_files, "mmsd_train_box.pt")
        img_val = os.path.join(img_files, "mmsd_val_box.pt")
        img_test = os.path.join(img_files, "mmsd_test_box.pt")
        img_edge_train = os.path.join(img_files, "mmsd_train_edge.pt")
        img_edge_val = os.path.join(img_files, "mmsd_val_edge.pt")
        img_edge_test = os.path.join(img_files, "mmsd_test_edge.pt")
        # img_train = os.path.join(img_files, "train_152.pt")
        # img_val = os.path.join(img_files, "val_152.pt")
        # img_test = os.path.join(img_files, "test_152.pt")
        train_dataset = BaseDataset(type="train", max_length=100, text_path=annotation_train,
                                use_np=use_np, img_path=img_train, edge_path=img_edge_train,
                                knowledge=0)
        val_dataset = BaseDataset(type="val", max_length=100, text_path=annotation_val, use_np=use_np,
                              img_path=img_val, edge_path=img_edge_val, knowledge=0)
        test_dataset = BaseDataset(type="test", max_length=100, text_path=annotation_test,
                               use_np=use_np, img_path=img_test, edge_path=img_edge_test, knowledge=0)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, num_workers=0,
                                      shuffle=True,
                                      collate_fn=PadCollate_without_know())
        print("training dataset has been loaded successful!")
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.train_batch_size, num_workers=0,
                                    shuffle=False,
                                    collate_fn=PadCollate_without_know())
        print("validation dataset has been loaded successful!")
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.train_batch_size, num_workers=0,
                                     shuffle=False,
                                     collate_fn=PadCollate_without_know())
        print("test dataset has been loaded successful!")


        processor = CLIPProcessor.from_pretrained("/mnt/afs/real_cyz/zzc/clip-vit-base-patch32")
        model = MIDL()
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    exit()
    train(args, model, device, train_loader, val_loader, test_loader, processor)



if __name__ == '__main__':
    main()
