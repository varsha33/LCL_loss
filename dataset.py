import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset

from collate_fns import collate_fn_emo, collate_fn_w_aug_emo, collate_fn_w_aug_sentiment, collate_fn_sentiment


class Emo_dataset(Dataset):

    def __init__(self, data, training=True, w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug

    def __getitem__(self, index):

        item = {'idx': [2*index, 2*index+1]}

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

        item = {'idx': [2*index, 2*index+1]}

        if self.training and self.w_aug:
            item["review"] = self.data["tokenized_review"][index]
        else:
            item["review"] = torch.LongTensor(self.data["tokenized_review"][index])

        item["sentiment"] = self.data["sentiment"][index]

        return item

    def __len__(self):
        return len(self.data["sentiment"])


def get_dataloader(batch_size, dataset, seed=None, w_aug=True, label_list=None):
    ## if using subset, change this part to input the correct data file
    if w_aug:
        with open('./preprocessed_data/' + dataset + '_waug_preprocessed_bert.pkl', "rb") as f:
            data = pickle.load(f)
        f.close()
    else:
        with open('./preprocessed_data/' + dataset + '_preprocessed_bert.pkl', "rb") as f:
            data = pickle.load(f)
        f.close()

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
