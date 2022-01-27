import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset

import lookup
from collate_fns import collate_fn_emo, collate_fn_w_aug_emo, collate_fn_w_aug_sentiment, collate_fn_sentiment


class Emo_dataset(Dataset):

    def __init__(self, data, training=True, w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug

    def __getitem__(self, index):

        item = {'idx': [2 * index, 2 * index + 1]}

        if self.training and self.w_aug:
            item["cause"] = self.data["tokenized_cause"][index]
        else:
            item["cause"] = torch.LongTensor(self.data["tokenized_cause"][index])

        item["emotion"] = self.data["emotion"][index]

        return item

    def __len__(self):
        return len(self.data["emotion"])


class SST_dataset(Dataset):

    def __init__(self, data, training=True, w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug

    def __getitem__(self, index):

        item = {'idx': [2 * index, 2 * index + 1]}

        if self.training and self.w_aug:
            item["review"] = self.data["tokenized_review"][index]
        else:
            item["review"] = torch.LongTensor(self.data["tokenized_review"][index])

        item["sentiment"] = self.data["sentiment"][index]

        return item

    def __len__(self):
        return len(self.data["sentiment"])


def get_dataloader(batch_size, dataset, seed=None, w_aug=True, label_list=None):
    with open('./preprocessed_data/' + dataset + f'_{"waug_" if w_aug else ""}preprocessed_bert.pkl', "rb") as f:
        data = pickle.load(f)
    if label_list and dataset == "ed":
        label_emo_set = set(map(lambda label: lookup.ed_label_dict[label], label_list))
        # label_emo_list = list(label_emo_set)
        for dataset_type in data:
            emotions = data[dataset_type]["emotion"]
            filtered_indices = [idx for idx in range(len(emotions)) if
                                (emotions[idx][0] if type(emotions[idx]) is list else emotions[idx]) in label_emo_set]
            for key, lists in data[dataset_type].items():
                data[dataset_type][key] = [data[dataset_type][key][i] for i in filtered_indices]
            # data[dataset_type]["emotion"] = [
            #     [label_emo_list.index(old_emo[0]), label_emo_list.index(old_emo[1])] if type(
            #         old_emo) is list else label_emo_list.index(old_emo) for old_emo in data[dataset_type]["emotion"]]

        # if using subset, change this part to input the correct data file

        if "sst-" in dataset:
            train_dataset = SST_dataset(data["train"], training=True, w_aug=w_aug)
            valid_dataset = SST_dataset(data["dev"], training=False, w_aug=w_aug)
            test_dataset = SST_dataset(data["test"], training=False, w_aug=w_aug)
        else:
            train_dataset = Emo_dataset(data["train"], training=True, w_aug=w_aug)
            valid_dataset = Emo_dataset(data["valid"], training=False, w_aug=w_aug)
            test_dataset = Emo_dataset(data["test"], training=False, w_aug=w_aug)

        if "sst-" in dataset:
            collate_fn = collate_fn_sentiment
            collate_fn_w_aug = collate_fn_w_aug_sentiment
        else:
            collate_fn = collate_fn_emo
            collate_fn_w_aug = collate_fn_w_aug_emo

        if w_aug:
            train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                     collate_fn=collate_fn_w_aug, num_workers=0)
        else:
            train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                     collate_fn=collate_fn, num_workers=0)

        valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                                 num_workers=0)

        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                                num_workers=0)

        return train_iter, valid_iter, test_iter
