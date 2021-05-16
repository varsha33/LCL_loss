import torch
import itertools


def collate_fn_emo(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: ## pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    data.sort(key=lambda x: len(x["cause"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    cause_batch,cause_attn_mask, cause_lengths = merge(item_info['cause'])

    d={}
    d["emotion"] = item_info["emotion"]
    d["cause"] = cause_batch
    d["cause_attn_mask"] = cause_attn_mask

    return d

def collate_fn_w_aug_emo(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: ## pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths


    ## each data sample has two views
    data.sort(key=lambda x: max(len(x["cause"][0]),len(x["cause"][1])), reverse=True) ## sort all the seq incl augmented

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
        ## unbinding the two views here as both views needs to be within the batch
        flat = itertools.chain.from_iterable(item_info[key])
        item_info[key] = list(flat)


    ## input
    cause_batch,cause_attn_mask, cause_lengths = merge(item_info['cause'])

    d={}

    d["emotion"] = item_info["emotion"]
    d["cause"] = cause_batch
    d["cause_attn_mask"] = cause_attn_mask

    return d

def collate_fn_sentiment(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: ## pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    data.sort(key=lambda x: len(x["review"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    review_batch,review_attn_mask, review_lengths = merge(item_info['review'])

    d={}
    d["sentiment"] = item_info["sentiment"]
    d["review"] = review_batch
    d["review_attn_mask"] = review_attn_mask

    return d

def collate_fn_w_aug_sentiment(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: ## pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths


    ## each data sample has two views
    data.sort(key=lambda x: max(len(x["review"][0]),len(x["review"][1])), reverse=True) ## sort all the seq incl augmented

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
        ## unbinding the two views here as both views needs to be within the batch
        flat = itertools.chain.from_iterable(item_info[key])
        item_info[key] = list(flat)


    ## input
    review_batch,review_attn_mask, review_lengths = merge(item_info['review'])

    d={}

    d["sentiment"] = item_info["sentiment"]
    d["review"] = review_batch
    d["review_attn_mask"] = review_attn_mask

    return d
