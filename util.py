import torch

def load_model(resume,model=None): ## for now model will be None for CVAE

    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    return model


def iter_product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def save_checkpoint(state,filename):
    torch.save(state,filename)


def clip_gradient(model, clip_value):

    for name,param in model.named_parameters():
        param.grad.data.clamp_(-clip_value, clip_value)

def one_hot(labels, class_size):
    if type(labels) is list:
      targets = torch.zeros(len(labels), class_size)
    else:
      targets = torch.zeros(labels.size(0), class_size)

    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets
