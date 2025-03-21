# MIDL

## Disentangled Multi-View Incongruity Learning for Multimodal Sarcasm Detection
![model](model.pdf)

Multimodal Sarcasm Detection (MSD) is a crucial task for understanding complex human communication and building intelligent emotional computing systems. However, existing MSD methods commonly suffer from an over-reliance on spurious correlations, causing the learned features to deviate from the true semantics of sarcasm. This bias severely compromises the generalization ability of current models beyond the training environment. This paper proposes a novel method that integrates **M**ultimodal **I**ncongruities through **D**isentangled **L**earning (MIDL), aiming to effectively disentangles and interacts with heterogeneous features in multimodal sarcasm. Considering that multimodal embedding spaces are typically heterogeneous, direct fusion will disrupt the inherent structure of embeddings from different modalities. To address this issue, we employ the optimal transport algorithm to align embeddings from different modalities into a unified space. Subsequently, we jointly learn incongruities from three views: token-patch, entity-object, and sentiment. To achieve debiased fusion of sarcasm features, we design a credibility-based fusion module to integrate the features from these three views. Experimental results demonstrate the superiority of MIDL on multimodal sarcasm datasets, and analysis shows that MIDL can effectively reduce reliance on spurious correlations. Additionally, out-of-distribution (OOD) experiments reveal that MIDL exhibits better generalization performance.


## Usage
python3 main.py --name mmsd --model MIDL --text_name text_json_clean --weight_decay 0.08 --train_batch_size 128 --dev_batch_size 128 --learning_rate 6e-4 --num_train_epochs 20 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0
