from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove final FC layer
        self.resnet.fc = nn.Identity()

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze ONLY layer4
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Projection to embedding space
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)   # [batch, 2048]

        features = self.linear(features)
        features = self.bn(features)

        return features