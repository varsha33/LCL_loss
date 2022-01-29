import argparse
import json
import os
import pickle
import time
from pathlib import Path
from easydict import EasyDict as edict

import config as train_config
from dataset import get_dataloader
from grid_search import grid_search
from train_new import lcl_train

## search space
search_space = {
    'main_learning_rate': [1e-5, 2e-5, 3e-5],
    'temperature'       : [0.1, 0.3, 0.5],
    'lambda_loss'       : [0.1, 0.2, 0.3, 0.4, 0.5],
    'alpha'             : [0.0, 0.3, 0.6, 0.9, 0.99],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ed', choices=['ed', 'emoint', 'goemotions', 'isear', 'sst-2', 'sst-5'])
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=250)
    parser.add_argument("--prev_results", type=str, default='')
    args = parser.parse_args()

    tuning_param = train_config.tuning_param
    seeds = train_config.SEED

    param = train_config.get_param(args.dataset)
    param['run_name'] = args.run_name
    for k, v in param.items():
        if isinstance(v, list):
            param[k] = v[0]

    log = edict()
    log.param = param

    data_loaders = get_dataloader(log.param.batch_size, log.param.dataset, w_aug=True, label_list=log.param.label_list)

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if log.param.run_name != "":
        save_home = "./save/search/" + log.param.dataset + "/" + log.param.run_name + "-" + log.param.loss_type + "/" + model_run_time + "/"
    else:
        save_home = "./save/search/" + log.param.dataset + "/" + log.param.loss_type + "/" + model_run_time + "/"

    if args.prev_results != '':
        prev_results = json.load(open(args.prev_results, 'r'))
    else:
        prev_results = None

    log.param.SEED = seeds

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

    study = grid_search(
        log=log,
        save_home=save_home,
        search_space=search_space,
        data_loaders=data_loaders,
        train_fn=lcl_train,
        metric='valid_emo_accuracy_1',
        n_repeats=args.n_repeats,
        n_trials=args.n_trials,
        prune_threshold=0.05,
        prev_results=prev_results
    )

    os.makedirs(save_home, exist_ok=True)
    pickle.dump(study.trials, open(f'{save_home}/search_res.pkl', 'wb'))

    best_param = study.best_params
    json.dump(best_param, open(f'{save_home}/best_param.json', 'w'))

    print(f'[BEST PARAM]: {best_param}')
    log.param.update(best_param)

    for seed in seeds:
        log.param.SEED = seed
        lcl_train(log, data_loaders=data_loaders, save_home=save_home)

    a = 1
