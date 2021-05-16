import pandas as pd

## Label mappings

ed_label_dict = {'Surprised': 0, 'Excited': 1, 'Annoyed': 2, 'Proud': 3, 'Angry': 4, 'Sad': 5, 'Grateful': 6, 'Lonely': 7,
'Impressed': 8, 'Afraid': 9, 'Disgusted': 10, 'Confident': 11, 'Terrified': 12, 'Hopeful': 13, 'Anxious': 14, 'Disappointed': 15,
'Joyful': 16, 'Prepared': 17, 'Guilty': 18, 'Furious': 19, 'Nostalgic': 20, 'Jealous': 21, 'Anticipating': 22, 'Embarrassed': 23,
'Content': 24, 'Devastated': 25, 'Sentimental': 26, 'Caring': 27, 'Trusting': 28, 'Ashamed': 29, 'Apprehensive': 30, 'Faithful': 31}

ed_emo_dict =  {v: k for k, v in ed_label_dict.items()}


isear_label_dict = {"joy":0,"fear":1,"anger":2,"sadness":3,"disgust":4,"shame":5,"guilt":6}

isear_emo_dict = {v: k for k, v in isear_label_dict.items()}

emoint_label_dict = {"anger":0,"fear":1,"joy":2,"sadness":3}

emoint_emo_dict = {v: k for k, v in emoint_label_dict.items()}

goemotions_label_dict= {"admiration":0,"amusement":1,"anger":2, "annoyance":3,"approval":4,"caring":5,"confusion":6,"curiosity":7,"desire":8,"disappointment":9,"disapproval":10,"disgust":11,"embarrassment":12,"excitement":13,"fear":14,"gratitude":15,"grief":16,"joy":17,"love":18,"nervousness":19,"optimism":20,"pride":21,"realization":22,"relief":23,"remorse":24,"sadness":25,"surprise":26}

goemotions_emo_dict = {v: k for k, v in goemotions_label_dict.items()}


target_names_label_dict = {"ed":ed_label_dict,"emoint":emoint_label_dict,"goemotions":goemotions_label_dict,"isear":isear_label_dict}
target_names_emo_dict = {"ed":ed_emo_dict,"emoint":emoint_emo_dict,"goemotions":goemotions_emo_dict,"isear":isear_emo_dict}
