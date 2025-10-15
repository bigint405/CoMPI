import torchvision
import torch.nn as nn
import torchvision.models as models


def resnet50(num_classes):
    resnet = models.resnet50(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet


def resnet101(num_classes):
    resnet = models.resnet101(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet


def resnet152(num_classes):
    resnet = models.resnet152(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet


def get_original_model_from_config(model_config):
    if model_config["type"].startswith("resnet"):
        resnet = getattr(models, model_config["type"])(pretrained=True)
        number_features = resnet.fc.in_features
        resnet.fc = nn.Linear(number_features, model_config["class"])
        return resnet


def get_model(name:str):
    if name.startswith("resnet") or name.startswith('vgg'):
        model = getattr(models, name)(weights=None)
    else:
        return None
    return model
    