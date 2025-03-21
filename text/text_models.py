import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from utils import L2_norm, cosine_distance
from transformers import BertModel
from utils.data_utils import pad_tensor
from transformers import RobertaModel
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig

class TextEncoder_without_know(nn.Module):
    r"""Initializes a NLP embedding block.
     :param input_size:
     :param nhead:
     :param dim_feedforward:
     :param dropout:
     :param activation:
     参数没有初始化
     """

    def __init__(self, input_size=512, out_size = 300):
        super(TextEncoder_without_know, self).__init__()

        self.input_size = input_size
        self.out_size = out_size


        self.norm = nn.LayerNorm(self.out_size)
        self.linear = nn.Linear(self.out_size, 1)
        self.linear_ = nn.Linear(self.input_size, self.out_size)
        self.config = RobertaConfig.from_pretrained('/mnt/afs/real_cyz/zzc/roberta-base')
        # self.bert_model = RobertaModel.from_pretrained('/data3/guodiandian/model/roberta-base')
            
    def get_config(self):
        return self.config

    def forward(self, t1, word_seq, key_padding_mask):
        """
        Function to compute forward pass of the ImageEncoder TextEncoder
        Args:
            t1: (N,L,D) Padded Tensor. L is the length. D is dimension after bert
            token_length: (N) list of length
            word_seq:(N,tensor) list of tensor
            key_padding_mask: (N,L1) Tensor. L1 is the np length. True means mask
        Returns:
            t1: (N,L1,D). The embedding of each word or np. D is dimension after bert.
            score: (N,L1,D). The importance of each word or np. For convenience, expand the tensor (N,L1，D) to compute
            the caption embedding.
        """
        #(batch_size, sequence_length, hidden_size)
        # output = t1
        # t1 = output[0]
        # print('t1:',t1.size())
        use = t1[:,0]
#         t1 = self.bert_model(**t1)[0]
        # (batch_size, hidden_size) may averaging or pooling
        cls_token = t1[:,0,:]
        # concatenate to np and words
        t1 = t1[:,1:-1,:]
        # print('t1:',t1.size())
        # captions = []
        # for i in range(t1.size(0)):
        #     # [X,L,H] X is the number of np and word
        #     captions.append(torch.stack([torch.mean(t1[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]]))
        captions = []
        for i in range(t1.size(0)):  # 遍历每个样本
            caption = []  # 当前样本的词语表示
            for tup in word_seq[i]:  # 遍历当前样本中每个词语的范围
                # 限制索引范围，确保不会超出t1的大小
                start_idx = min(tup[0], t1.size(1) - 1)  # 确保start_idx不超过序列长度
                end_idx = min(tup[1], t1.size(1))  # 确保end_idx不超过序列长度
                
                # 获取当前词语的嵌入
                word_embedding = t1[i][start_idx:end_idx, :]
                
                # 检查该切片是否为空
                if word_embedding.size(0) > 0:
                    # 计算均值
                    caption.append(torch.mean(word_embedding, dim=0))
                # else:
                #     # 如果为空，可以选择跳过、填充默认值或其他处理方法
                #     caption.append(torch.zeros_like(word_embedding[0]))  # 用零填充，可以根据需要选择其他默认值
            
            # 将处理后的词语表示按顺序堆叠
            captions.append(torch.stack(caption))
        # print('cap:',len(captions))
        # print('cap:',captions[0].size())
        # (N,L,D)
        t1 = pad_sequence(captions, batch_first=True).cuda()
        t1 = self.norm(self.linear_(t1))

        return t1,use
