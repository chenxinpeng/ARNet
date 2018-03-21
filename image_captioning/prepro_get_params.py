import torch

def get_params(model):
    params = 0
    for k, v in model.items():
        if len(v.size()) == 1:
            params += v.size()[0]
        if len(v.size()) == 2:
            params += v.size()[0] * v.size()[1]
    return params


if __name__ == "__main__":
    model = torch.load("")
    print(get_params(model))
