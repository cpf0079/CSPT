from torchvision import models
import torch


def select_resnet(network):
    param = {'feature_size': 2048}
    if network == 'resnet18':
        model = models.resnet18(pretrained=False)
        new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        param['feature_size'] = 512
    elif network == 'resnet34':
        model = models.resnet34(pretrained=False)
        new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        param['feature_size'] = 512
    elif network == 'resnet50':
        model = models.resnet50(pretrained=False)
        new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif network == 'resnet101':
        model = models.resnet101(pretrained=False)
        new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif network == 'resnet152':
        model = models.resnet152(pretrained=False)
        new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    else: raise IOError('model type is wrong')

    return new_model, param


if __name__ == "__main__":
    model = models.resnet50(pretrained=False)
    new_model = torch.nn.Sequential(*(list(model.children())[:-1]))

