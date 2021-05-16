import re
from lookup import emoint_label_dict
def parse(data_file,dataset):
	"""
	Taken from https://github.com/cbaziotis/ntua-slp-semeval2018.git
	Returns:
		X: a list of tweets
		y: a list of lists corresponding to the emotion labels of the tweets
	"""

	if dataset == "emoint":
		with open(data_file, 'r') as fd:
			data = [l.strip().split('\t') for l in fd.readlines()][1:]

		X = [d[1] for d in data]

		y = [d[2] for d in data]

	return X, y

def preprocess(data_file,dataset):
	## taken from https://github.com/abdulfatir/twitter-sentiment-analysis.git

	X,y = parse(data_file,dataset)

	X_proc,y_proc = [],[]
	count = 0
	for i,x_i in enumerate(X):
		# print(i)

	# 	## remove non_ascii like emojis etc
		x_proc_i= [''.join([i if ord(i) < 128 else '' for i in text]) for text in x_i]
		x_proc_i = "".join(x_proc_i).replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')

		##
		## <https... or www... --> URL
		x_proc_i = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ',x_proc_i)

		## @<handle> --> USER_MENTION
		x_proc_i = re.sub(r'@[\S]+', 'USER_MENTION', x_proc_i)

		## &amp; --> and
		x_proc_i = x_proc_i.replace(r'&amp;',r'and')

		## #<hastag> --> <hastag>
		x_proc_i = re.sub(r'#(\S+)', r' \1 ', x_proc_i)

		## remove rt --> space
		x_proc_i = re.sub(r'\brt\b', '', x_proc_i)

		## remove more than 2 dots (..) --> space
		x_proc_i = re.sub(r'\.{2,}', ' ', x_proc_i)

		x_proc_i = x_proc_i.strip(' "\'')

		## remove multiple space with single space
		x_proc_i = re.sub(r'\s+', ' ', x_proc_i)

		y_proc.append(emoint_label_dict[y[i]])
		X_proc.append(x_proc_i)

	return X_proc,y_proc
