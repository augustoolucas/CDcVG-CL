import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Specific(nn.Module):
    def __init__(self, img_shape, specific_n_hidden):
        super(Specific, self).__init__()

        channels = img_shape[0] if img_shape[0] < img_shape[2] else img_shape[2]
        # specific module
        self.specific = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), specific_n_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        x = self.specific(imgs.view(imgs.shape[0], -1))
        return x

     
class Classifier(nn.Module):
    def __init__(self, invariant_n_hidden, specific_n_hidden, classification_n_hidden, n_classes):
        super(Classifier, self).__init__()

        # classification module
        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_n_hidden + invariant_n_hidden, classification_n_hidden),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Linear(classification_n_hidden, n_classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, discriminative, invariant):
        x = self.classifier_layer(torch.cat([discriminative, invariant], dim=1))
        logits = self.output(x)
        return logits
