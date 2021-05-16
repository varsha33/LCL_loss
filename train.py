import pandas as pd
import numpy as np
import json
import random
import os
import sys
import pickle
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import config as train_config
from dataset import get_dataloader
from util import save_checkpoint,one_hot,iter_product,clip_gradient,load_model
from sklearn.metrics import accuracy_score,f1_score
import loss as loss
from model import primary_encoder,weighting_network
import math

from transformers import AdamW,get_linear_schedule_with_warmup

def train(epoch,train_loader,model_main,model_helper,loss_function,optimizer,lr_scheduler,log):

    model_main.cuda()
    model_main.train()

    model_helper.cuda()
    model_helper.train()

    total_emo_true,total_emo_pred_1,total_emo_pred_2,acc_curve_1,acc_curve_2 = [],[],[],[],[]
    train_loss_1,train_loss_2 = 0,0
    total_epoch_acc_1,total_epoch_acc_2 = 0,0
    steps = 0
    start_train_time = time.time()

    train_batch_size = log.param.batch_size*2
    for idx,batch in enumerate(train_loader):

        if "sst-" in log.param.dataset:
            text_name = "review"
            label_name = "sentiment"
        else:
            text_name = "cause"
            label_name = "emotion"

        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        emotion = batch[label_name]
        emotion = torch.tensor(emotion)
        emotion = torch.autograd.Variable(emotion).long()


        if (emotion.size()[0] is not train_batch_size):# Last batch may have length different than log.param.batch_size
            continue

        if torch.cuda.is_available():

            text = text.cuda()
            attn = attn.cuda()
            emotion = emotion.cuda()

        # class_weights,emo_pred,supcon_feature= model(text,attn)
        emo_pred_1,supcon_feature_1 = model_main(text,attn)
        emo_pred_2= model_helper(text,attn)

        loss_1 = (loss_function["lambda_loss"]*loss_function["emotion"](emo_pred_1,emotion)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive"](supcon_feature_1,emotion,emo_pred_2))
        loss_2 = loss_function["lambda_loss"]*loss_function["emotion"](emo_pred_2,emotion)

        loss = loss_1 + loss_2
        train_loss_1  += loss_1.item()
        train_loss_2  += loss_2.item()


        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(model_helper.parameters(), max_norm=1.0)
        optimizer.step()
        model_main.zero_grad()
        model_helper.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print (f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.4f},Training Loss_2: {loss_2.item():.4f}, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        emo_true_list = emotion.data.detach().cpu().tolist()
        total_emo_true.extend(emo_true_list)

        num_corrects_1 = (torch.max(emo_pred_1, 1)[1].view(emotion.size()).data == emotion.data).float().sum()

        emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].view(emotion.size()).data.detach().cpu().tolist()

        total_emo_pred_1.extend(emo_pred_list_1)


        acc_1 = 100.0 * (num_corrects_1/train_batch_size)
        acc_curve_1.append(acc_1.item())
        total_epoch_acc_1 += acc_1.item()

        num_corrects_2 = (torch.max(emo_pred_2, 1)[1].view(emotion.size()).data == emotion.data).float().sum()

        emo_pred_list_2 = torch.max(emo_pred_2, 1)[1].view(emotion.size()).data.detach().cpu().tolist()

        total_emo_pred_2.extend(emo_pred_list_2)

        acc_2 = 100.0 * (num_corrects_2/train_batch_size)
        acc_curve_2.append(acc_2.item())
        total_epoch_acc_2 += acc_2.item()


    return train_loss_1/len(train_loader),train_loss_2/len(train_loader),total_epoch_acc_1/len(train_loader),total_epoch_acc_2/len(train_loader),acc_curve_1,acc_curve_2


def test(epoch,test_loader,model_main,model_helper,loss_function,log):
    model_main.eval()
    model_helper.eval()
    test_loss = 0
    total_epoch_acc_1,total_epoch_acc_2 = 0,0
    total_emo_pred_1,total_emo_pred_2,total_emo_true,total_pred_prob_1,total_pred_prob_2 = [],[],[],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_2":[],"pred_prob_1":[],
    "pred_prob_2":[],"feature":[]}
    acc_curve_1,acc_curve_2 = [],[]
    total_feature = []
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):

            if "sst-" in log.param.dataset:
                text_name = "review"
                label_name = "sentiment"
            else:
                text_name = "cause"
                label_name = "emotion"

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            emotion = batch[label_name]
            emotion = torch.tensor(emotion)
            emotion = torch.autograd.Variable(emotion).long()

            if torch.cuda.is_available():

                text = text.cuda()
                attn = attn.cuda()
                emotion = emotion.cuda()

            emo_pred_1,supcon_feature_1= model_main(text,attn)
            emo_pred_2 = model_helper(text,attn)

            num_corrects_1 = (torch.max(emo_pred_1, 1)[1].view(emotion.size()).data == emotion.data).float().sum()

            emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].view(emotion.size()).data.detach().cpu().tolist()
            emo_true_list = emotion.data.detach().cpu().tolist()


            acc_1 = 100.0 * num_corrects_1/1
            acc_curve_1.append(acc_1.item())
            total_epoch_acc_1 += acc_1.item()

            num_corrects_2 = (torch.max(emo_pred_2, 1)[1].view(emotion.size()).data == emotion.data).float().sum()
            emo_pred_list_2 = torch.max(emo_pred_2, 1)[1].view(emotion.size()).data.detach().cpu().tolist()

            total_emo_pred_1.extend(emo_pred_list_1)
            total_emo_pred_2.extend(emo_pred_list_2)
            total_emo_true.extend(emo_true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(emo_pred_1.data.detach().cpu().tolist())
            total_pred_prob_2.extend(emo_pred_2.data.detach().cpu().tolist())

            acc_2 = 100.0 * num_corrects_2/1
            acc_curve_2.append(acc_2.item())
            total_epoch_acc_2 += acc_2.item()

    f1_score_emo_1 = f1_score(total_emo_true,total_emo_pred_1, average="macro")
    f1_score_emo_2 = f1_score(total_emo_true,total_emo_pred_2, average="macro")

    f1_score_emo_1_w = f1_score(total_emo_true,total_emo_pred_1, average="weighted")
    f1_score_emo_2_w = f1_score(total_emo_true,total_emo_pred_2, average="weighted")

    f1_score_1 = {"macro":f1_score_emo_1,"weighted":f1_score_emo_1_w}
    f1_score_2 = {"macro":f1_score_emo_2,"weighted":f1_score_emo_2_w}

    save_pred["true"] = total_emo_true
    save_pred["pred_1"] = total_emo_pred_1
    save_pred["pred_2"] = total_emo_pred_2
    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1
    save_pred["pred_prob_2"] = total_pred_prob_2

    return total_epoch_acc_1/len(test_loader),total_epoch_acc_2/len(test_loader),f1_score_1,f1_score_2,save_pred,acc_curve_1,acc_curve_2

def lcl_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    train_data,valid_data,test_data = get_dataloader(log.param.batch_size,log.param.dataset,w_aug=True,label_list=log.param.label_list)


    losses = {"contrastive":loss.LCL(temperature=log.param.temperature),"emotion":nn.CrossEntropyLoss(),"lambda_loss":log.param.lambda_loss}


    train_loss_overall,test_loss_overall,train_emo_accuracy_overall,test_emo_accuracy_overall = 0,0,0,0


    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_main = primary_encoder(log.param.batch_size,log.param.hidden_size,log.param.emotion_size,log.param.model_type)

    model_helper = weighting_network(log.param.batch_size,log.param.hidden_size,log.param.emotion_size,log.param.model_type)


    total_params = list(model_main.named_parameters()) + list(model_helper.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    if log.param.run_name != "":
        save_home = "./save/final/"+log.param.dataset+"/"+log.param.run_name+"/"+log.param.loss_type+"/"+model_run_time+"/"
    else:
        save_home = "./save/final/"+log.param.dataset+"/"+log.param.loss_type+"/"+model_run_time+"/"

    total_train_acc_curve_1,total_train_acc_curve_2,total_val_acc_curve_1,total_val_acc_curve_2 = [],[],[],[]

    for epoch in range(1, log.param.nepoch + 1):

        train_loss_1,train_loss_2,train_acc_1,train_acc_2,train_acc_curve_1,train_acc_curve_2 = train(epoch,train_data,model_main,model_helper,losses,optimizer,lr_scheduler,log)

        val_acc_1,val_acc_2,val_f1_1,val_f1_2,val_save_pred,val_acc_curve_1,val_acc_curve_2 = test(epoch,valid_data,model_main,model_helper,losses,log)

        test_acc_1,test_acc_2,test_f1_1,test_f1_2,test_save_pred,test_acc_curve_1,test_acc_curve_2 = test(epoch,test_data,model_main,model_helper,losses,log)

        total_train_acc_curve_1.extend(train_acc_curve_1)
        total_val_acc_curve_1.extend(val_acc_curve_1)

        total_train_acc_curve_2.extend(train_acc_curve_2)
        total_val_acc_curve_2.extend(val_acc_curve_2)

        # lr_scheduler_1.step()

        print('====> Epoch: {} Train loss_1: {:.4f} Train loss_2: {:.4f}'.format(epoch, train_loss_1,train_loss_2))

        os.makedirs(save_home,exist_ok=True)
        with open(save_home+"/acc_curve.json", 'w') as fp:
            json.dump({"train_acc_curve_1":total_train_acc_curve_1,"train_acc_curve_2":total_train_acc_curve_2,"val_acc_curve_1":total_val_acc_curve_1,"val_acc_curve_2":total_val_acc_curve_2}, fp,indent=4)
        fp.close()

        if epoch == 1:
             best_criterion = 0

        is_best = val_acc_1 > best_criterion
        best_criterion = max(val_acc_1,best_criterion)

        print("Model 1")
        print(f'Valid Accuracy: {val_acc_1:.2f} Emotion Valid F1: {val_f1_1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_1:.2f} Emotion Test F1: {test_f1_1["macro"]:.2f}')

        print("Model 2")
        print(f'Valid Accuracy: {val_acc_2:.2f} Emotion Valid F1: {val_f1_2["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_2:.2f} Emotion Test F1: {test_f1_2["macro"]:.2f}')

        if is_best:
            print("======> Best epoch <======")
            patience_flag = 0
            log.train_loss_1 = train_loss_1
            log.train_loss_2 = train_loss_2
            log.stop_epoch = epoch
            log.stop_step = len(total_val_acc_curve_2)
            log.valid_emo_f1_score_1 = val_f1_1
            log.test_emo_f1_score_1 = test_f1_1
            log.valid_emo_accuracy_1 = val_acc_1
            log.test_emo_accuracy_1 = test_acc_1
            log.train_emo_accuracy_1 = train_acc_1

            log.valid_emo_f1_score_2 = val_f1_2
            log.test_emo_f1_score_2 = test_f1_2
            log.valid_emo_accuracy_2 = val_acc_2
            log.test_emo_accuracy_2 = test_acc_2
            log.train_emo_accuracy_2 = train_acc_2

            ## load the model
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp,indent=4)
            fp.close()

            with open(save_home+"/feature.json", 'w') as fp:
                json.dump(test_save_pred, fp,indent=4)
            fp.close()

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val
        if log.param.run_name == "subset":
            log.param.emotion_size = int(log.param.label_list.split("-")[0])
        ## reseeding before every run while tuning

        if log.param.dataset == "ed":
            log.param.emotion_size = 32
        elif log.param.dataset == "emoint":
            log.param.emotion_size = 4
        elif log.param.dataset == "goemotions":
            log.param.emotion_size = 27
        elif log.param.dataset == "isear":
            log.param.emotion_size = 7
        elif log.param.dataset == "sst-2":
            log.param.emotion_size = 2
        elif log.param.dataset == "sst-5":
            log.param.emotion_size = 5

        lcl_train(log)


