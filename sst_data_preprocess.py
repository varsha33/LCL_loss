import pandas as pd
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
import random
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
## torch packages
import torch
from transformers import BertTokenizer,AutoTokenizer


## custom packages
from lookup import ed_emo_dict,ss_label_dict,isear_label_dict,goemotions_label_dict,goemotions_emo_dict,enisear_label_dict,emotion_stimulus_label_dict
import tweet

aug = naw.SynonymAug(aug_src='wordnet')

def preprocess_data(dataset,tokenizer_type,w_aug):

        print("Extracting data")
        data_home = "./data/sst/"+dataset+"/"

        data_dict = {}
        for datatype in ["train","dev","test"]:

            if datatype == "train" and w_aug:
                data = pd.read_csv(data_home+datatype+".csv")
                final_review,final_sentiment = [],[]

                for i,val in enumerate(data["label"]):

                    final_review.append(data["sentence"][i])
                    final_sentiment.append(val)


                augmented_review = aug.augment(final_review)

                print("Tokenizing data")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                tokenized_review_original =tokenizer.batch_encode_plus(final_review).input_ids
                tokenized_review_augmented =tokenizer.batch_encode_plus(augmented_review).input_ids

                tokenized_combined_review = [list(i) for i in zip(tokenized_review_original,tokenized_review_augmented)]
                combined_review = [list(i) for i in zip(final_review,augmented_review)]
                combined_sentiment = [list(i) for i in zip(final_sentiment,final_sentiment)]

                processed_data = {}


                # ## changed review --> review for uniformity
                processed_data["tokenized_review"] = tokenized_combined_review
                processed_data["sentiment"] = combined_sentiment
                processed_data["review"] = combined_review

                data_dict[datatype] = processed_data

            else:
                data = pd.read_csv(data_home+datatype+".csv")
                final_review,final_sentiment = [],[]

                for i,val in enumerate(data["label"]):

                    final_review.append(data["sentence"][i])
                    final_sentiment.append(val)

                print("Tokenizing data")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                tokenized_review_original =tokenizer.batch_encode_plus(final_review).input_ids

                processed_data = {}

                # ## changed review --> review for uniformity
                processed_data["tokenized_review"] = tokenized_review_original
                processed_data["sentiment"] = final_sentiment
                processed_data["review"] = final_review

                data_dict[datatype] = processed_data


            if w_aug:
                with open("./preprocessed_data/"+dataset+"_waug_preprocessed_bert.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
                f.close()
            else:
                with open("./preprocessed_data/"+dataset+"_preprocessed_bert.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
                f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Enter tokenizer type')

    parser.add_argument('-d', default="sst-5",type=str,
                   help='Enter dataset=')
    parser.add_argument('-t', default="bert-base-uncased",type=str,
                   help='Enter tokenizer type')
    parser.add_argument('--aug', action='store_true')

    args = parser.parse_args()


    preprocess_data(args.d,args.t,w_aug=args.aug)
