import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from easydict import EasyDict as edict
from sklearn.metrics import f1_score
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

import config as train_config
from dataset import get_dataloader
from loss import SupConLoss
from model import primary_encoder
from util import iter_product


def train(epoch, train_loader, model_main, loss_function, optimizer, lr_scheduler, log):
    model_main.cuda()
    model_main.train()

    total_emo_true, total_emo_pred_1, acc_curve_1 = [], [], []
    train_loss_1 = 0
    total_epoch_acc_1 = 0
    steps = 0
    start_train_time = time.time()

    train_batch_size = log.param.batch_size * 2
    for idx, batch in enumerate(train_loader):

        if "sst-" in log.param.dataset:
            text_name = "review"
            label_name = "sentiment"
        else:
            text_name = "cause"
            label_name = "emotion"

        text = batch[text_name]
        attn = batch[text_name + "_attn_mask"]
        emotion = batch[label_name]
        emotion = torch.tensor(emotion)
        emotion = torch.autograd.Variable(emotion).long()

        if emotion.size()[0] is not train_batch_size:  # Last batch may have length different from log.param.batch_size
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            emotion = emotion.cuda()

        # class_weights,emo_pred,supcon_feature= model(text,attn)
        emo_pred_1, supcon_feature_1 = model_main(text, attn)
        if type(loss_function) is nn.CrossEntropyLoss:
            loss_1 = loss_function(emo_pred_1, emotion)
        elif type(loss_function) is SupConLoss:
            loss_1 = loss_function(supcon_feature_1, emotion)
        else:
            raise ValueError("Only cross entropy and supervised contrastive loss are supported.")

        train_loss_1 += loss_1.item()

        loss_1.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)
        optimizer.step()
        model_main.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(
                f'Epoch: {epoch:02}, Idx: {idx + 1}, Training Loss_1: {loss_1.item():.4f}, Time taken: {((time.time() - start_train_time) / 60): .2f} min')
            start_train_time = time.time()

        emo_true_list = emotion.data.detach().cpu().tolist()
        total_emo_true.extend(emo_true_list)

        num_corrects_1 = (torch.max(emo_pred_1, 1)[1].view(emotion.size()).data == emotion.data).float().sum()

        emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].view(emotion.size()).data.detach().cpu().tolist()

        total_emo_pred_1.extend(emo_pred_list_1)

        acc_1 = 100.0 * (num_corrects_1 / train_batch_size)
        acc_curve_1.append(acc_1.item())
        total_epoch_acc_1 += acc_1.item()
    return train_loss_1 / len(train_loader), total_epoch_acc_1 / len(train_loader), acc_curve_1


def test(test_loader, model_main, log):
    model_main.eval()
    total_epoch_acc_1 = 0
    total_emo_pred_1, total_emo_true, total_pred_prob_1 = [], [], []
    save_pred = {"true": [], "pred_1": [], "pred_prob_1": [], "feature": []}
    acc_curve_1 = []
    total_feature = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):

            if "sst-" in log.param.dataset:
                text_name = "review"
                label_name = "sentiment"
            else:
                text_name = "cause"
                label_name = "emotion"

            text = batch[text_name]
            attn = batch[text_name + "_attn_mask"]
            emotion = batch[label_name]
            emotion = torch.tensor(emotion)
            emotion = torch.autograd.Variable(emotion).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                emotion = emotion.cuda()

            emo_pred_1, supcon_feature_1 = model_main(text, attn)

            num_corrects_1 = (torch.max(emo_pred_1, 1)[1].view(emotion.size()).data == emotion.data).float().sum()

            emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].view(emotion.size()).data.detach().cpu().tolist()
            emo_true_list = emotion.data.detach().cpu().tolist()

            acc_1 = 100.0 * num_corrects_1 / 1
            acc_curve_1.append(acc_1.item())
            total_epoch_acc_1 += acc_1.item()

            total_emo_pred_1.extend(emo_pred_list_1)
            total_emo_true.extend(emo_true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(emo_pred_1.data.detach().cpu().tolist())

    f1_score_emo_1 = 100 * f1_score(total_emo_true, total_emo_pred_1, average="macro")

    f1_score_emo_1_w = 100 * f1_score(total_emo_true, total_emo_pred_1, average="weighted")

    f1_score_1 = {"macro": f1_score_emo_1, "weighted": f1_score_emo_1_w}

    save_pred["true"] = total_emo_true
    save_pred["pred_1"] = total_emo_pred_1
    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_epoch_acc_1 / len(test_loader), f1_score_1, save_pred, acc_curve_1


def lcl_train(log, loss_for_emotion, data_loaders=None, save_home=None, test_flag=True):
    print(f"loss function is {str(loss_for_emotion)}")
    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    if save_home is not None:
        save_home = save_home + "/" + str(log.param.SEED)

    if data_loaders is None:
        train_data, valid_data, test_data = get_dataloader(log.param.batch_size, log.param.dataset, w_aug=True,
                                                           label_list=log.param.label_list)
    else:
        train_data, valid_data, test_data = data_loaders

    model_main = primary_encoder(log.param.batch_size, log.param.hidden_size, log.param.emotion_size,
                                 log.param.model_type)

    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data) * log.param.nepoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)],
         'weight_decay': log.param.decay},
        {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    total_train_acc_curve_1, total_val_acc_curve_1 = [], []

    best_criterion = 0

    for epoch in range(1, log.param.nepoch + 1):

        train_loss_1, train_acc_1, train_acc_curve_1, = train(epoch,
                                                              train_data,
                                                              model_main,
                                                              loss_for_emotion,
                                                              optimizer,
                                                              lr_scheduler,
                                                              log)

        val_acc_1, val_f1_1, val_save_pred, val_acc_curve_1, = test(valid_data,
                                                                    model_main,
                                                                    log)

        total_train_acc_curve_1.extend(train_acc_curve_1)
        total_val_acc_curve_1.extend(val_acc_curve_1)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))

        if save_home is not None:
            os.makedirs(save_home, exist_ok=True)
            with open(save_home + "/acc_curve.json", 'w') as fp:
                json.dump({"train_acc_curve_1": total_train_acc_curve_1, "val_acc_curve_1": total_val_acc_curve_1}, fp,
                          indent=4)
        is_best = val_acc_1 > best_criterion
        best_criterion = val_acc_1 if is_best else best_criterion

        print(f'Valid Accuracy: {val_acc_1:.2f} Emotion Valid F1: {val_f1_1["macro"]:.2f}')

        if is_best:
            print("======> Best epoch <======")
            log.train_loss_1 = train_loss_1
            log.stop_epoch = epoch
            log.stop_step = len(total_val_acc_curve_1)

            log.train_emo_accuracy_1 = train_acc_1
            log.valid_emo_f1_score_1 = val_f1_1
            log.valid_emo_accuracy_1 = val_acc_1

            if test_flag:
                test_acc_1, test_f1_1, test_save_pred, test_acc_curve_1, = test(test_data,
                                                                                model_main,
                                                                                log)
                print(f'Test Accuracy: {test_acc_1:.2f} Emotion Test F1: {test_f1_1["macro"]:.2f}')

                log.test_emo_f1_score_1 = test_f1_1
                log.test_emo_accuracy_1 = test_acc_1

            # load the model
            if save_home is not None:
                with open(save_home + "/log.json", 'w') as fp:
                    json.dump(dict(log), fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ed',
                        choices=['ed', 'emoint', 'goemotions', 'isear', 'sst-2', 'sst-5'])
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument('--label_list', type=str, nargs='+', default=[])
    parser.add_argument("--loss_function", type=str, default="scl", choices=["ce", "scl"])
    args = parser.parse_args()

    tuning_param = train_config.tuning_param
    seeds = train_config.SEED

    param = train_config.get_param(args.dataset)
    if len(args.label_list) > 0:
        param['label_list'] = args.label_list
    param['run_name'] = args.run_name
    param['loss_type'] = args.loss_function

    param_list = [param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list))  ## [(param_name),(param combinations)]
    log = edict()
    log.param = param

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if log.param.run_name != "":
        save_home = "./save/final/" + log.param.dataset + "/" + log.param.run_name + "/" + log.param.loss_type + "/" + model_run_time + "/"
    else:
        save_home = "./save/final/" + log.param.dataset + "/" + log.param.loss_type + "/" + model_run_time + "/"

    if log.param.run_name == "subset":
        log.param.emotion_size = int(log.param.label_list.split("-")[0])
    ## reseeding before every run while tuning

    if log.param.dataset == "ed":
        log.param.emotion_size = len(log.param.label_list) if log.param.label_list else 32
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
    for param_com in param_list[1:]:  # as first element is just name
        for num, val in enumerate(param_com):
            log.param[param_list[0][num]] = val

    for seed in seeds:
        log.param.SEED = seed
        lcl_train(log, nn.CrossEntropyLoss() if log.param.loss_type == "ce" else SupConLoss(),
                  save_home=save_home)
        print(f"seed {seed} finished")
        print()
