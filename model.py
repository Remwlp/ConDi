from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import torch.nn.functional as F
from train import clip_model

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values

class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("/data3/guodiandian/model/clip")
        self.config = BertConfig.from_pretrained("/data3/guodiandian/model/bert-base")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        # self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        # self.classifier_text = nn.Linear(args.text_size, args.label_number)
        # self.classifier_image = nn.Linear(args.image_size, args.label_number)
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)


        self.att = nn.Linear(args.text_size, 1, bias=False)
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True)
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        cross_feature_text = self.cross_att(text_feature, image_feature, image_feature)  # 32,768
        cross_feature_image = self.cross_att(image_feature, text_feature, text_feature)  # 32,768
        fuse_feature = 0.7 * cross_feature_text + 0.3 * cross_feature_image
        output = fuse_feature

        logits_fuse = self.classifier_fuse(output)  # 64,2  output
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)  # 64,2

        score = fuse_score

        outputs = (score,) # (64,2)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss = loss_fuse
            outputs = (loss,) + outputs
        return outputs
        # text_embeds = self.model.text_projection(text_features)
        # image_embeds = self.model.visual_projection(image_features)
        # input_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        # attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        # fuse_hiddens = fuse_hiddens[-1]
        # new_text_features = fuse_hiddens[:, 50:, :]
        # new_text_feature = new_text_features[
        #     torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        # ]

        # new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        # text_weight = self.att(new_text_feature)
        # image_weight = self.att(new_image_feature)    
        # att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        # tw, iw = att.split([1,1], dim=-1)
        # fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        # logits_fuse = self.classifier_fuse(fuse_feature)
        # logits_text = self.classifier_text(text_feature)
        # logits_image = self.classifier_image(image_feature)
   
        # fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        # text_score = nn.functional.softmax(logits_text, dim=-1)
        # image_score = nn.functional.softmax(logits_image, dim=-1)

        # score = fuse_score + text_score + image_score

        # outputs = (score,)
        # if labels is not None:
        #     loss_fuse = self.loss_fct(logits_fuse, labels)
        #     loss_text = self.loss_fct(logits_text, labels)
        #     loss_image = self.loss_fct(logits_image, labels)
        #     loss = loss_fuse + loss_text + loss_image

        #     outputs = (loss,) + outputs
        # return outputs


import math
import torch.nn as nn
import torch
from images.image_models import ImageEncoder
from text.text_models import TextEncoder_without_know
from interraction.inter_models import CroModality
import utils.gat as tg_conv
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig, AutoTokenizer
from utils import L2_norm, cosine_distance
from transformers import BertModel
from utils.data_utils import pad_tensor
from utils.pre_model import RobertaEncoder,BertCrossEncoder,BertPooler,ProjetTransformer,CrossTransformer
import copy

def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

def sinkhorn(b, a, C, reg=1e-1, method='sinkhorn', maxIter=100, tau=1e3,
             stopThr=1e-9, verbose=False, log=True, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)

def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):


    C=C.t()
    device = a.device
    na, nb = C.shape


    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'


    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb


    K = torch.empty(C.shape, dtype=C.dtype).to(device)

    K1=-reg*C
    K=torch.exp(K1)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1


    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    M_EPS = torch.tensor(1e-16).to(device)
    while (err > stopThr and it <= maxIter):

        u=u.to(device);v=v.to(device);b=b.to(device);K=K.to(device)
        upre, vpre = u, v

        KTu=torch.matmul(u.to(torch.float32), K.to(torch.float32))
        v = torch.div(b, KTu + M_EPS)
        Kv=torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:

            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()

            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v

        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)


    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


class MLP_DISENTANGLE(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DISENTANGLE, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, num_classes)

    def extract(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        return feat

    # def predict(self, x):
    #     prediction = self.classifier(x)
    #     return prediction

    # def forward(self, x, mode=None, return_feat=False):
    #     x = x.view(x.size(0), -1) / 255
    #     feat = x = self.feature(x)
    #     final_x = self.classifier(x)
    #     if mode == 'tsne' or mode == 'mixup':
    #         return x, final_x
    #     else:
    #         if return_feat:
    #             return final_x, feat
    #         else:
    #             return final_x

class Text_Graph_Encoder(nn.Module):
    def __init__(self, input_size=300, txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False):
        super(Text_Graph_Encoder, self).__init__()

        self.input_size = input_size
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.GELU()

        self.txt_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])

    def forward(self, t2, edge_index, gnn_mask):
        # (N,token_length)
        tnp = t2
        # for node with out edge, it representation will be zero-vector
        
        for gat in self.txt_conv:
            tnp = self.norm(torch.stack([(self.relu1(gat(data[0], data[1].cuda(), mask=data[2]))) for data in zip(tnp, edge_index, gnn_mask)]))
        #  congruity score of compositional level

        return tnp

class Prompt_Encoder(nn.Module):
    def __init__(self, input_size=300, img_gat_layer=2, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False):
        super(Prompt_Encoder, self).__init__()
        self.input_size = input_size
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.img_conv = nn.ModuleList([tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size,
                                                        heads=self.img_gat_head, concat=False,
                                                        dropout=self.img_gat_drop, fill_value="mean",
                                                        add_self_loops=self.img_self_loops, is_text=True) for i in
                                          range(self.img_gat_layer)])
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.GELU()

#         # for token compute the importance of each token
#         self.linear1 = nn.Linear(self.input_size, 1)
#         # for np compute the importance of each np
#         self.linear2 = nn.Linear(self.input_size, 1)
#         self.norm = nn.LayerNorm(self.input_size)
#         self.relu1 = nn.ReLU()

    def forward(self, v2, img_edge_index):
        # prompt graph encoder
        v3 = v2
        for gat in self.img_conv:
            v3 = self.norm(torch.stack([self.relu1(gat(data[0], data[1].cuda())) for data in zip(v3, img_edge_index)]))

        return v3
    

import torch
from torch import nn
import ot  # 使用POT库来进行最优传输计算
import torch
from torch import nn
import ot  # 使用POT库来进行最优传输计算

class OptimalTransportAlignment(nn.Module):
    def __init__(self, reg=1e-2, max_iter=200):
        super(OptimalTransportAlignment, self).__init__()
        self.reg = reg  # 最优传输正则化系数
        self.max_iter = max_iter  # Sinkhorn算法最大迭代次数

    def forward(self, A, B):
        """
        输入:
            A: 模态A的特征，形状为 [batch_size, 768]，位于GPU
            B: 模态B的特征，形状为 [batch_size, 768]，位于GPU
        输出:
            对齐后的A特征和B特征，形状为 [batch_size, 768]，位于GPU
        """
        # 计算A和B之间的距离矩阵
        dist_matrix = self.compute_distance_matrix(A, B)
        dist_matrix = torch.clamp(dist_matrix, min=1e-6)

        # 计算最优传输矩阵
        transport_matrix = self.compute_transport_matrix(dist_matrix)

        # 根据最优传输矩阵对A和B进行对齐
        aligned_A = self.apply_transport(A, transport_matrix)
        aligned_B = self.apply_transport(B, transport_matrix.T)  # 对B进行转置的对齐

        return aligned_A, aligned_B

    def compute_distance_matrix(self, A, B):
        """
        计算模态A和模态B之间的欧几里得距离矩阵。
        """
        # 计算A和B的每一对特征向量之间的欧几里得距离
        dist_matrix = torch.cdist(A, B)  # 计算A和B之间的距离矩阵，形状 [batch_size, batch_size]
        dist_matrix = torch.clamp(dist_matrix, min=1e-6)  # 防止距离矩阵包含非常小的数值
        dist_matrix = dist_matrix / dist_matrix.max()  # 标准化距离矩阵
        return dist_matrix

    def compute_transport_matrix(self, dist_matrix):
        """
        计算最优传输矩阵。使用Sinkhorn算法来计算。
        """
        # 将dist_matrix转换为numpy格式，以便与POT库兼容
        dist_matrix_cpu = dist_matrix.detach().cpu().numpy()  # 必须先转移到CPU才能使用ot.sinkhorn

        # Sinkhorn算法计算最优传输矩阵
        transport_matrix = ot.sinkhorn(
            torch.ones(dist_matrix.size(0)).cpu().numpy(),  # 源分布的权重
            torch.ones(dist_matrix.size(0)).cpu().numpy(),  # 目标分布的权重
            dist_matrix_cpu,  # 距离矩阵需要转换为numpy格式
            reg=self.reg,  # Sinkhorn正则化参数
            numItermax=self.max_iter,  # 最大迭代次数
            stopThr=1e-5,  # 收敛阈值
            log=False,  
        )

        # 转换为PyTorch张量，并迁移回GPU
        transport_matrix = torch.tensor(transport_matrix, dtype=torch.float32).to(dist_matrix.device)
        return transport_matrix

    def apply_transport(self, X, transport_matrix):
        """
        根据最优传输矩阵对特征X进行对齐。
        """
        # 将X的特征进行对齐
        aligned_X = torch.matmul(transport_matrix, X)
        return aligned_X



class MIDL(nn.Module):
    """
    Our model for Image Repurpose Task
    """

    def __init__(self, txt_input_dim=512, txt_out_size=150, img_input_dim=768, img_inter_dim=500, img_out_dim=150,
                 cro_layers=6, cro_heads=5, cro_drop=0.2,
                 txt_gat_layer=1, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=1, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, img_edge_dim=0,
                 img_patch=49, lam=1, type_bmco=0, rank = 500, visualization=False):
        super(MIDL, self).__init__()
        self.rank = rank
        self.scale = 768
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size

        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim

        if self.img_out_dim is not self.txt_out_size:
            self.img_out_dim = self.txt_out_size

        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.type_bmco = type_bmco

        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.img_edge_dim = img_edge_dim

        if self.img_gat_layer is not self.txt_gat_layer:
            self.img_gat_layer = self.txt_gat_layer
        if self.img_gat_drop is not self.txt_gat_drop:
            self.img_gat_drop = self.txt_gat_drop
        if self.img_gat_head is not self.txt_gat_head:
            self.img_gat_head = self.txt_gat_head

        # self.img_patch = img_patch
        
        self.txt_encoder = TextEncoder_without_know(input_size=self.txt_input_dim, out_size=self.txt_out_size)


        self.text_tokenizer = AutoTokenizer.from_pretrained('/mnt/afs/real_cyz/zzc/roberta-base',add_prefix_space=True)
        
        self.text_config = copy.deepcopy(self.txt_encoder.get_config())    
        self.text_config.num_attention_heads = self.cro_heads
        self.text_config.hidden_size = self.txt_out_size
        self.text_config.num_hidden_layers = 1
        
        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False
        
        
        self.img_encoder = ImageEncoder(input_dim=self.img_input_dim, inter_dim=self.img_inter_dim,
                                        output_dim=self.img_out_dim)
        self.img_prompt = Prompt_Encoder(input_size=self.img_out_dim, img_gat_layer=2
                                   , img_gat_drop=self.img_gat_drop, img_gat_head=self.img_gat_head,
                                   img_self_loops=self.img_self_loops)
        self.text_prompt_encoder = RobertaEncoder(self.text_config)    
        
        self.text_graph_encoder = Text_Graph_Encoder(input_size=self.img_out_dim, txt_gat_layer=self.txt_gat_layer,
                                   txt_gat_drop=self.txt_gat_drop,
                                   txt_gat_head=self.txt_gat_head, txt_self_loops=self.txt_self_loops)
        self.output_attention = nn.Linear(self.img_out_dim, 1)
    



        self.e1 = nn.Sequential(nn.Linear(in_features=768, out_features=2),nn.Softmax(dim=-1))
        self.e2 = nn.Sequential(nn.Linear(in_features=100, out_features=2),nn.Softmax(dim=-1))
        self.e3 = nn.Sequential(nn.Linear(in_features=img_out_dim, out_features=2),nn.Softmax(dim=-1))


        ## optimal transport
        self.ot = OptimalTransportAlignment()

        # ## disentangled augmentation
        self.model_p = ProjetTransformer(num_frames=129, dim=768, depth=1, heads=8, mlp_dim=512, dim_head = 64, len_invariant = 8, len_specific = 8, dropout = 0.2, emb_dropout = 0.1)
        
        
        
        self.text_linear =  nn.Sequential(
                nn.Linear(512, img_input_dim),
                nn.Dropout(img_gat_drop),
                nn.GELU()
        )
        self.image_linear =  nn.Sequential(
                nn.Linear(img_input_dim, img_input_dim),
                nn.Dropout(img_gat_drop),
                nn.GELU()
        )
        
        
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)




    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq, txt_edge_index,
                gnn_mask, np_mask, clip_input, img_edge_attr=None, key_padding_mask_img=None, twitters=None):
        
        output = clip_model(**clip_input,output_attentions=True)
        tt = output['text_model_output']['pooler_output']
        ii = output['vision_model_output']['pooler_output']
        txt = output['text_model_output']['last_hidden_state']
        imm = output['vision_model_output']['last_hidden_state']
        tt = self.text_linear(tt)
        ii = self.image_linear(ii)
        
        tt,ii=self.ot(tt,ii)
        txt_feature = self.text_linear(txt)
        img_feature = self.image_linear(imm)
        
        
        


        mul_feature = torch.cat((tt.unsqueeze(1),txt_feature,ii.unsqueeze(1),img_feature),dim=1) 
        # mul_feature = torch.cat((txt_feature,img_feature),dim=1) 
        separate_feature = self.model_p(mul_feature)

        x_invariant, x_specific_t, x_specific_v = separate_feature[:, :8], separate_feature[:, 8:12], separate_feature[:, 12:16]
        
        feat_specific = torch.cat((x_specific_t, x_specific_v), dim=1)
        feat_1 = self.cross_att(x_invariant, feat_specific, feat_specific).mean(dim=1)
        feat_2 = self.cross_att(feat_specific, x_invariant, x_invariant).mean(dim=1)
        
        
        # cross_feature_text = self.cross_att(tt, ii, ii)  # 32,768
        # cross_feature_image = self.cross_att(ii, tt, tt)  # 32,768
        # word_patch_incongruity = 0.5*cross_feature_text + 0.5*cross_feature_image 
         

        word_patch_incongruity = feat_1 + feat_2





   



        senti_feature = get_text_sentiment(twitters, self.text_tokenizer, imgs.device)







        imgs, __ = self.img_encoder(imgs)
        texts, __ = self.txt_encoder(t1=txt, word_seq=t1_word_seq,
                                        key_padding_mask=mask_batch)
        
        imgs_prompt = self.img_prompt(imgs, img_edge_index=img_edge_index)
        prompt_mask = torch.ones(imgs_prompt.size()[:-1],dtype=torch.long).to(imgs.device)
        t1 = self.text_graph_encoder(texts, edge_index=txt_edge_index, gnn_mask=gnn_mask)
        mask_batch = mask_batch.long()
        ones_mask = torch.ones(mask_batch.size(),dtype=torch.long).to(imgs.device)
        mask_batch = torch.abs(mask_batch-ones_mask)
        text_image_cat = torch.cat([imgs_prompt, t1], dim=1)
        text_image_mask = torch.cat([prompt_mask, mask_batch], dim=1)
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_image_cat.size())

        text_transformer = self.text_prompt_encoder(text_image_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_transformer = text_transformer.last_hidden_state
        
        text_mask = text_image_mask.permute(1, 0).contiguous()
        text_mask = text_mask[0:text_transformer.size(1)]
        text_mask = text_mask.permute(1, 0).contiguous()
        text_image_alpha = self.output_attention(text_transformer)
        text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_mask == 0, -1e9)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
    
        text_image_output = (text_image_alpha.unsqueeze(-1) * text_transformer).sum(dim=1)




        e1 = self.e1(word_patch_incongruity)
        e2 = self.e2(senti_feature)
        e3 = self.e3(text_image_output)
        evidences = [e3,e2,e1]
        evidence_a = evidences[0]

        for i in range(1, 3):
            evidence_a = (evidences[i] + evidence_a) / 2



        return evidence_a, tt, ii, x_invariant, x_specific_t, x_specific_v , evidences



import nltk


from senticnet.senticnet import SenticNet
import numpy as np
import torch


sn = SenticNet()





def get_text_sentiment(texts, tokenizer, device):
    res = []
    for text in texts:
        if tokenizer is not None: # 使用 RoBERTa tokenizer
            encoded_input = tokenizer(text, is_split_into_words=True, return_tensors="pt", truncation=True,
                                     max_length=100,  padding='max_length')
            tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
            words = text
        else: words = text.split()
        # word_sentiments={}
        sentiment_scores = []
        current_word = ""
        current_sentiment = 0.0
        for token in tokens:
            if token.startswith("Ġ"):  # 新词的开始
                current_word = token[1:]  # 去除 RoBERTa 特有的前缀
                if current_word in words:
                    try:
                        current_sentiment = float(sn.concept(current_word)['polarity_value'])
                    except:
                        current_sentiment = 0.0
                else:
                    current_sentiment = 0.0  # 如果当前词不在words列表中，使用默认情感值0
            elif current_word:
                # 处理 subword，这些通常是一个词的一部分
                current_word += token.replace("Ġ", "")
                try:
                    current_sentiment = float(sn.concept(current_word)['polarity_value'])
                except:
                    current_sentiment = 0.0
            # try:
            #     current_sentiment = float(sn.concept(token)['polarity_value'])
            # except:
                # current_sentiment = float(0)
            sentiment_scores.append(current_sentiment)
            
        res.append(torch.tensor(sentiment_scores).to(device))
        # res.append(sentiment_scores)

    res = torch.tensor(np.array([item.cpu().detach().numpy() for item in res])).to(device)

    return res


def beta_weight(e,feature):
    # 计算S，首先分离出e_0和e_1
    e_0 = e[:, 0]
    e_1 = e[:, 1]

    # 根据给定公式计算S
    S = (e_0 + e_1) / (e_0 + 1 + e_1 + 1)
    S = S.view(-1, 1)  # 改变形状以匹配[64, 1]
    return feature * S, S