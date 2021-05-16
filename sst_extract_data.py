## Credits https://github.com/prrao87/fine-grained-sentiment/blob/master/data/sst/tree2tabular.py
# Load data
import re
import pytreebank
import sys
import os
import pandas as pd

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string

out_path ="./data/sst/sst-5/"
dataset = pytreebank.load_sst('./data/sst/raw_data') ## download the split version from the standford nlp page

# Store train, dev and test in separate files

for category in ['train', 'test', 'dev']:
    label,sentence = [],[]
    outfile = out_path+category+".csv"
    for item in dataset[category]:
        label.append(item.to_labeled_lines()[0][0])
        sentence.append(clean_str_sst(item.to_labeled_lines()[0][1]))
    df = pd.DataFrame({"label":label,"sentence":sentence})
    df.to_csv(outfile)

out_path ="./data/sst/sst-2/"
dataset = pytreebank.load_sst('./data/sst/raw_data')

# Store train, dev and test in separate files
for category in ['train', 'test', 'dev']:
    label,sentence = [],[]
    outfile = out_path+category+".csv"
    for item in dataset[category]:
        label_i = item.to_labeled_lines()[0][0]
        if label_i == 2:
            continue
        else:
            if label_i < 2:
                label.append(0)
            elif label_i > 2:
                label.append(1)
            sentence.append(clean_str_sst(item.to_labeled_lines()[0][1]))

    df = pd.DataFrame({"label":label,"sentence":sentence})
    df.to_csv(outfile)


