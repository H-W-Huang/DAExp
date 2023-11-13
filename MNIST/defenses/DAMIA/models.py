import torch.nn as nn
import torchvision
import torch.nn.functional as F
# from Coral import CORAL
import mmd
import backbone
import numpy as np



class C100_classifier(nn.Module):
    def __init__(self, input_shape):
        super(C100_classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( input_shape , 1024),
            nn.ReLU(),
            nn.Linear( 1024 , 512),
            nn.ReLU(),
            nn.Linear( 512 , 256),
            nn.ReLU(),
            nn.Linear( 256 , 128),
            nn.Linear( 128 , 100),
            nn.Softmax(dim =1)
        )

    def forward(self, x):
        pred = self.model(x)
        return pred



class Attacker_model(nn.Module):
    def __init__(self, input_shape, cls_num = 2):
        super(Attacker_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( input_shape , 512),
            nn.ReLU(),
            nn.Linear( 512 , 256),
            nn.ReLU(),
            nn.Linear( 256 , 128),
            nn.Linear( 128 , cls_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.model(x)
        return pred






class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='alexnet', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target) 
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        return source_clf, transfer_loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss