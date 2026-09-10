# 当前复现状态

PHEME 指定实验已完成；Weibo 忠实复现仍受外部知识资源阻塞。

| 实验 | 已完成折数 | Accuracy 均值 ± 样本标准差 | Macro-F1 均值 ± 样本标准差 | 论文 Acc / F1 |
|---|---:|---:|---:|---:|
| PHEME DDGCN | 5/5 | 0.852298 ± 0.008771 | 0.840072 ± 0.011072 | 0.855 / 0.844 |
| PHEME w/o TF | 5/5 | 0.851415 ± 0.021519 | 0.837996 ± 0.023245 | 0.842 / 0.833 |
| PHEME w/o dynamic | 5/5 | 0.852637 ± 0.017245 | 0.840309 ± 0.018020 | 0.841 / 0.835 |
| PHEME w/o knowledge | 5/5 | 0.852471 ± 0.010640 | 0.839754 ± 0.013154 | 0.833 / 0.819 |
| PHEME w/o propagation | 5/5 | 0.854902 ± 0.012635 | 0.842737 ± 0.015417 | 0.832 / 0.821 |
| Weibo 忠实复现 | 0/5，知识资源阻塞 | — | — | 0.948 / 0.950 |

PHEME 主模型权重与预测位于 `/root/autodl-tmp/ddgcn-runs/pheme_cv_checkpointed`，已通过五折全新进程重载核验，不应重训或覆盖。原始 Weibo 已完整下载，剩余依赖是论文使用的实体链接和 Probase 知识资源。

完整命令、环境、哈希、early detection 表格及修改说明见 [REPRODUCTION.md](REPRODUCTION.md)，阶段状态见 [PAPER_PROGRESS.md](PAPER_PROGRESS.md)，Weibo 外部依赖见 [WEIBO_RESOURCE_STATUS.md](WEIBO_RESOURCE_STATUS.md)。当前无需安装依赖或重新下载原始数据。

下方为作者原始说明，其中旧环境及作者本地 wheel 路径不适用于当前服务器。

# DDGCN
Code for the AAAI 2022 paper "[DDGCN: Dual Dynamic Graph Convolutional Networks for Rumor Detection on Social Media](https://www.aaai.org/AAAI22Papers/AAAI-6370.SunM.pdf)"



Make sure the following files are present as per the directory structure before running the code,
```
DDGCN
├── README.md
├── preprocess
|   ├──  *.py
|   └──  *.txt
├── models
|   ├── *.py 
|   └── model_saved
|        └── ckpt_nn.model
├── data
|   └── pheme
|        ├── all-rnr-annotated-threads
|        │   ├── ebola-essien-all-rnr-threads
|        │   ├── charliehebdo-all-rnr-threads
|            ├── ......
|        │   └── sydneysiege-all-rnr-threads
|        ├── pheme_clean
|        ├── pheme_concept_yago
|        ├── pheme_entity
|        ├── pheme_temporal_data
|        ├── mid2text.txt
|        ├── mid2token.csv
|        ├── node2idx.txt
|        ├── node2idx_mid.txt
|        ├── node2idx_test.txt
|        └── pheme_id_label.txt
└── requirement.txt

```
pheme_clean, pheme_concept_yago, pheme_entity and pheme_temporal_data these four folders are packed into a zip file, and can be obtained from https://www.dropbox.com/s/xwn5dvqgx2n2vsd/pheme_peocessed_data.zip?dl=0.
The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 (or https://www.dropbox.com/s/j8x105s60ow997f/all-rnr-annotated-threads.zip?dl=0).

The raw Weibo dataset can be downloaded from https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0 .  More information about this dataset is available here 'https://github.com/majingCUHK/Rumor_GAN'.



# Dependencies

Our code runs with the following packages installed:
```
python 3.6
torch 1.4.0+cu100
torch-cluster 1.5.2
torch-geometric 1.7.2
torch-scatter 2.0.3
torch-sparse 0.5.1
tqdm
numpy
pandas
matplotlib
scikit_learn
```

install the virtual environment with 
```
pip install -r requirement.txt
```



# Run

Train and test,
```
python train_dynamic.py --dataset pheme --model completed --cuda 1 --batch 32 --epoch 5 --lr 0.001
```


