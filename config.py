all_datasets = ['ed', 'emoint', 'goemotions', 'isear', 'sst-2', 'sst-5']

# search space
search_space = {
    'main_learning_rate': [1e-5, 2e-5, 3e-5],
    'temperature': [0.1, 0.3, 0.5],
    'lambda_loss': [0.1, 0.2, 0.3, 0.4, 0.5],
}

dataset = "sst-2"

# list of possible paramters to be tuned can be increased
tuning_param = ["lambda_loss", "main_learning_rate", "nepoch", "temperature"]

if dataset == 'goemotions':
    lambda_loss = [0.1]
else:
    lambda_loss = [0.5]

if dataset == 'sst-5':
    temperature = [0.1]
else:
    temperature = [0.3]

if dataset == 'isear':
    main_learning_rate = [3e-5]
else:
    main_learning_rate = [2e-5]

batch_size = 10
decay = 1e-02

hidden_size = 768
nepoch = [5]
criterion = "emo_acc"
run_name = ""  # for subset use "subset" else leave empty
loss_type = "lcl"
model_type = "electra"
label_list = ["Anticipating", "Excited", "Hopeful", "Guilty"]

SEED = [0, 1, 2, 3, 4]

param = {"temperature": temperature, "run_name": run_name, "dataset": dataset, "main_learning_rate": main_learning_rate,
         "batch_size": batch_size, "hidden_size": hidden_size, "nepoch": nepoch, "criterion": criterion,
         "lambda_loss": lambda_loss, "loss_type": loss_type, "label_list": label_list,
         "decay": decay, "model_type": model_type}


def get_param(dataset):
    if dataset == 'goemotions':
        lambda_loss = [0.1]
    else:
        lambda_loss = [0.5]

    if dataset == 'sst-5':
        temperature = [0.1]
    else:
        temperature = [0.3]

    if dataset == 'isear':
        main_learning_rate = [3e-5]
    else:
        main_learning_rate = [2e-5]

    param = {"temperature": temperature, "run_name": run_name, "dataset": dataset,
             "main_learning_rate": main_learning_rate,
             "batch_size": batch_size, "hidden_size": hidden_size, "nepoch": nepoch, "criterion": criterion,
             "lambda_loss": lambda_loss, "loss_type": loss_type, "label_list": label_list,
             "decay": decay, "model_type": model_type}

    return param
