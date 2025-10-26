import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str = "efficientnet_b0", num_classes: int = 2):
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
        return m
    elif model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    else:
        # Minimal fallback
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
