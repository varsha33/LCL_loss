import pandas as pd
import numpy as np
import json
import random
import os
import sys
import pickle
import itertools

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collate_fns import collate_fn_emo,collate_fn_w_aug_emo,collate_fn_w_aug_sentiment,collate_fn_sentiment

class Emo_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["cause"] = self.data["tokenized_cause"][index]
        else:
            item["cause"] = torch.LongTensor(self.data["tokenized_cause"][index])

        item["emotion"] = self.data["emotion"][index]

        return item

    def __len__(self):
        return len(self.data["emotion"])


class SST_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["review"] = self.data["tokenized_review"][index]
        else:
            item["review"] = torch.LongTensor(self.data["tokenized_review"][index])

        item["sentiment"] = self.data["sentiment"][index]

        return item

    def __len__(self):
        return len(self.data["sentiment"])

def get_dataloader(batch_size,dataset,seed=None,w_aug=True,label_list=None):

    if w_aug:
        if label_list == "16":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_16.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()

        elif label_list == "8":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_8.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "4-easy":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_4_easy.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "4-hard":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_4_hard.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()

        elif label_list == "4-hard1":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_4-hard1.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "3-hard":
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert_3-hard.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()

        else:
            with open('./preprocessed_data/'+dataset+'_waug_preprocessed_bert.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
    else:
        if label_list == "16":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_16.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()

        elif label_list == "8":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_8.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "4-easy":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_4_easy.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "4-hard":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_4_hard.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "4-hard1":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_4-hard1.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        elif label_list == "3-hard":
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert_3-hard.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()
        else:
            with open('./preprocessed_data/'+dataset+'_preprocessed_bert.pkl', "rb") as f:
                data = pickle.load(f)
            f.close()

    if "sst-" in dataset:
        train_dataset = SST_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = SST_dataset(data["dev"],training=False,w_aug=w_aug)
        test_dataset = SST_dataset(data["test"],training=False,w_aug=w_aug)
    else:
        train_dataset = Emo_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = Emo_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = Emo_dataset(data["test"],training=False,w_aug=w_aug)

    if "sst-" in dataset:
        collate_fn = collate_fn_sentiment
        collate_fn_w_aug = collate_fn_w_aug_sentiment
    else:
        collate_fn = collate_fn_emo
        collate_fn_w_aug = collate_fn_w_aug_emo

    if w_aug:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn_w_aug,num_workers=0)
    else:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)

    valid_iter  = torch.utils.data.DataLoader(valid_dataset, batch_size=1,shuffle=False,collate_fn=collate_fn,num_workers=0)

    test_iter  = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False,collate_fn=collate_fn,num_workers=0)


    return train_iter,valid_iter,test_iter

