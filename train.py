import os
from data_set import MyDataset
from utils.dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.compute_scores import get_metrics

from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained("/mnt/afs/real_cyz/zzc/clip-vit-base-patch32").cuda()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

# 定义自定义损失函数
class MultiViewDigammaLoss(nn.Module):
    def __init__(self):
        super(MultiViewDigammaLoss, self).__init__()

    def forward(self, x, y):
        # x 的 shape 为 [batch_size, 6]
        # 将 x 分解为三个 [batch_size, 2] 的 Tensor
        x1, x2, x3 = x[:, :2], x[:, 2:4], x[:, 4:]

        # 用于累加每个视角的损失
        total_loss = 0

        for xi in [x1, x2, x3]:
            # 使用 y 作为索引，从每个视角的 xi 中选择对应的元素
            indices = y.unsqueeze(1)  # 将 y 的 shape 从 [batch_size, 1] 变为 [batch_size, 1, 1]
            selected_x = torch.gather(xi, 1, indices)  # 选择 xi 中的元素
            selected_x = selected_x.squeeze(1)  # 移除多余的维度

            # 应用 digamma 函数
            digamma_values = torch.digamma(1 + selected_x)

            # 计算当前视角的损失
            loss = torch.sum(y * digamma_values)

            # 将当前视角的损失加到总损失上
            total_loss += loss

        return total_loss/len(y)

digmma_loss = MultiViewDigammaLoss()

import torch
import torch.nn.functional as F

def info_nce_loss(textual_features, visual_features, labels, temperature=0.07):
    """
    计算带标签的 advanced InfoNCE 损失，包含两部分：
    1. Textual features 与 Visual features 之间的 InfoNCE 损失。
    2. Visual features 与 Textual features 之间的 InfoNCE 损失。
    
    正样本是与当前样本具有相同标签的所有样本。
    
    参数:
    - textual_features: 文本特征矩阵，形状为 (batch_size, feature_dim)
    - visual_features: 视觉特征矩阵，形状为 (batch_size, feature_dim)
    - labels: 每个样本的标签，形状为 (batch_size,)
    - temperature: 温度系数，默认为 0.07
    
    返回:
    - 最终的平均 InfoNCE 损失
    """
    batch_size = textual_features.shape[0]

    # 计算 textual -> visual 余弦相似度矩阵 (batch_size, batch_size)
    sim_textual_to_visual = torch.mm(textual_features, visual_features.t()) / temperature
    # 计算 visual -> textual 余弦相似度矩阵 (batch_size, batch_size)
    sim_visual_to_textual = torch.mm(visual_features, textual_features.t()) / temperature

    # 创建正样本掩码矩阵 (batch_size, batch_size)，相同标签的位置为1，否则为0
    labels = labels.clone().detach().unsqueeze(1)  # 将标签转换为列向量
    positive_mask = torch.eq(labels, labels.t()).float()  # 标签相同的位置为1，其他位置为0

    # Textual -> Visual InfoNCE 损失
    exp_sim_textual_to_visual = torch.exp(sim_textual_to_visual)  # 对相似度取指数
    exp_sum_textual_to_visual = exp_sim_textual_to_visual.sum(dim=1)  # 对 batch 内所有 visual 特征求和

    # 正样本对的相似性是按标签匹配的
    sim_positive_textual_to_visual = (exp_sim_textual_to_visual * positive_mask).sum(dim=1)

    # Textual -> Visual 的 InfoNCE 损失
    loss_textual_to_visual = -torch.log(sim_positive_textual_to_visual / exp_sum_textual_to_visual)

    # Visual -> Textual InfoNCE 损失
    exp_sim_visual_to_textual = torch.exp(sim_visual_to_textual)  # 对相似度取指数
    exp_sum_visual_to_textual = exp_sim_visual_to_textual.sum(dim=1)  # 对 batch 内所有 textual 特征求和

    # 正样本对的相似性是按标签匹配的
    sim_positive_visual_to_textual = (exp_sim_visual_to_textual * positive_mask).sum(dim=1)

    # Visual -> Textual 的 InfoNCE 损失
    loss_visual_to_textual = -torch.log(sim_positive_visual_to_textual / exp_sum_visual_to_textual)

    # 最终损失是两个损失的平均
    total_loss = (loss_textual_to_visual.mean() + loss_visual_to_textual.mean()) / 2
    
    return total_loss

cross_entropy_loss = CrossEntropyLoss()

class OrthLoss(nn.Module):

    def __init__(self):
        super(OrthLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def conflict_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum    

def construct_edge_image_region(image):
    """
    Args:
        num_patches: the patches of image (49)
    There are two kinds of construct method
    Returns:
        edge_image(2,num_edges): List. num_edges = num_boxes*num_boxes
    """
    # fully connected 构建方法
    # for i in range(num_patches):
    #     edge_image.append(torch.stack([torch.full([num_patches], i, dtype=torch.long),
    #                                    torch.arange(num_patches, dtype=torch.long)]))
    # edge_image = torch.cat(edge_image, dim=1)
    # remove self-loop 
    image_region = image.size(1)
    image_batch = image.size(0)
    all_edge = []
    for k in range(image_batch):
        edge_image = []
        for i in range(image_region):
            for j in range(image_region):
                if j == i:
                    continue
                if F.cosine_similarity(image[k, i, :], image[k, j, :], dim=-1) > 0.6:
                    edge_image.append([i, j])
        edge_image = torch.tensor(edge_image, dtype=torch.long).T
        all_edge.append(edge_image)
    return all_edge
def train(args, model, device, train_loader, dev_loader, test_loader, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)
    orthloss = OrthLoss()
    train_loss = 0.0

    if args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()
        for step, batch in enumerate(iter_bar):
            img_batch, img_edge, embed_batch1, org_seq, org_word_len, mask_batch1, edge_cap1, gnn_mask_1, np_mask_1,  labels, key_padding_mask_img, target_labels,twitters,text_list,img_list = batch
            
            
            
            embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
            batch = len(img_batch)
            for i in range(len(target_labels)):
                target_labels[i] = target_labels[i].cuda()
    #             img_edge_index = construct_edge_image_region(img_batch) 
                
            
            
            inputs = processor(text=text_list, images=img_list, padding='max_length', truncation=True, max_length=77, return_tensors="pt").to(device)



            with torch.set_grad_enabled(True):
                y, cl_text, cl_visual, x_invariant, x_specific_t, x_specific_v, evidences = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                            img_edge_index=img_edge,
                            t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(), img_edge_attr=None, key_padding_mask_img=key_padding_mask_img,
                                                        twitters=twitters, clip_input=inputs)

                cl_text = F.normalize(cl_text, dim=-1)
                cl_visual = F.normalize(cl_visual, dim=-1)

                cl_loss = info_nce_loss(cl_text,cl_visual,labels.cuda())
                or_loss = (orthloss(x_invariant.mean(dim=1), x_specific_t.mean(dim=1)) + orthloss(x_invariant.mean(dim=1), x_specific_v.mean(dim=1)))/2 
                    
                co_loss = conflict_loss(evidences, device)
                #loss = cross_entropy_loss(y, labels.cuda()) + 0.2*cl_loss + 0.2*or_loss
                loss = cross_entropy_loss(y, labels.cuda()) + 0.2*cl_loss + 0.2*or_loss + co_loss
                train_loss += float(loss.detach().item())
                loss.backward()
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()  # clear gradients for this training step

        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_loader, processor, mode='dev')
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        if dev_acc > max_acc:
            max_acc = dev_acc

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model_'+args.name+'.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_loader, processor,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_loader, processor, mode='test')
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data_loader, processor, macro=False,pre = None, mode='test'):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                img_batch, img_edge, embed_batch1, org_seq, org_word_len, mask_batch1, edge_cap1, gnn_mask_1, np_mask_1,  labels, key_padding_mask_img, target_labels,twitters,text_list,img_list = t_batch
            
                inputs = processor(text=text_list, images=img_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                
                t_targets = labels.cuda()
                # loss, t_outputs = model(inputs,labels=labels)
                # sum_loss += loss.item()
                sum_step += 1
  
                # outputs = torch.argmax(t_outputs, -1)

                # n_correct += (outputs == t_targets).sum().item()
                # n_total += len(outputs)
                lam = 1
                embed_batch1 = {k: v.to(device) for k, v in embed_batch1.items()}
    #                 img_edge_index = construct_edge_image_region(img_batch)
                y, txt, vis , _,_,_,_ = model(imgs=img_batch.cuda(), texts=embed_batch1, mask_batch=mask_batch1.cuda(),
                                    img_edge_index=img_edge,
                                    t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                                    np_mask=np_mask_1.cuda(), img_edge_attr=None,
                                    key_padding_mask_img=key_padding_mask_img,
                                                        twitters=twitters, clip_input = inputs)
                predicted_labels = torch.tensor(get_metrics(y.cpu())).cuda()
                final_label = []
                final_label=predicted_labels
                loss = cross_entropy_loss(y, labels.cuda())
                sum_loss += loss.item()

                n_correct += (final_label == t_targets).sum().item()
                n_total += len(final_label)
                outputs = final_label

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall
