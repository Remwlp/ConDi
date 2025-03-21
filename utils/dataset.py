import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image   

class BaseDataset(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, img_path=None, edge_path=None, knowledge=0):
        """
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding. Resnet152(,2048), Vit B_32(,768), Vit L_32(, 1024)
            use_np: True or False, whether use noun phrase as relation matching node. It is useless in this paper.
            img_path:
            knowledge: 1 caption, 2 ANP, 3 attribute, 0 not use knowledge
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.egde_path = edge_path
        self.use_np = use_np
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.img_set = torch.load(self.img_path)
        self.edge_set = torch.load(self.egde_path)
        self.knowledge = int(knowledge)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            img: (49, 768). Tensor.
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        # if 'mmsd' not in self.text_path and 'reddit' not in  self.text_path:
        #     if self.type == "train":
        #         label = sample[2]
        #         text = sample[3]
        #     else:
        #         # label =sample[2] hashtag label
        #         label = sample[3]
        #         text = sample[4]
            
        #     if self.use_np:
        #         twitter = text["chunk_cap"]
        #         dep = text["chunk_dep"]
        #         chunk_index = text["chunk_index"]
        #     else:
        #         twitter = text["token_cap"]
        #         dep = text["token_dep"]
        # else:
        label = sample['label']
        text = sample['text']
        twitter = sample["chunk_cap"]
        dep = sample["chunk_dep"]
        if 'mmsd' in self.text_path or 'ood' in self.text_path:
            img_path = "./dataset_image/"+ str(sample["image_id"]) + '.jpg'
        elif 'reddit' in self.text_path:
            img_path = "./reddit/"+ str(sample["image_id"]) + '.jpg'
        # useless in this project
        img = self.img_set[index]
        edge = self.edge_set[index]
        if self.knowledge == 0:
            return img, edge, twitter, dep, label, text, img_path

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)

class BaseSet_2(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, img_path=None, edge_path=None, knowledge=0):
        """
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding. Resnet152(,2048), Vit B_32(,768), Vit L_32(, 1024)
            use_np: True or False, whether use noun phrase as relation matching node. It is useless in this paper.
            img_path:
            knowledge: 1 caption, 2 ANP, 3 attribute, 0 not use knowledge
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.egde_path = edge_path
        self.use_np = use_np
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.img_set = torch.load(self.img_path)
        self.edge_set = torch.load(self.egde_path)
        self.knowledge = int(knowledge)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            img: (49, 768). Tensor.
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        if 'mmsd' not in self.text_path:
            if self.type == "train":
                label = sample[2]
                text = sample[3]
            else:
                # label =sample[2] hashtag label
                label = sample[3]
                text = sample[4]
            
            if self.use_np:
                twitter = text["chunk_cap"]
                dep = text["chunk_dep"]
                chunk_index = text["chunk_index"]
            else:
                twitter = text["token_cap"]
                dep = text["token_dep"]
        else:
            label = sample['label']
            text = sample['text']
            twitter = sample["chunk_cap"]
            dep = sample["chunk_dep"]
            img = "/data3/guodiandian/dataset/twitter/dataset_image/"+ str(sample["image_id"]) + '.jpg'



        # img = self.img_set[index]
        edge = self.edge_set[index]
        if self.knowledge == 0:
            return img, edge, text, dep, label

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)
    
    
    
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        batches = []


        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for instance in batch_data:
            text_list.append(instance[2])
            image_list.append(Image.open(instance[0]))
            label_list.append(instance[4])
        return text_list, image_list, label_list
