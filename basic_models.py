import torch
import torch.nn as nn


class FCNN_dropout(nn.Module):
    def __init__(self,input_shape, cls_num):
        super(FCNN_dropout, self).__init__()
        self.fcnn = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256,cls_num)
        )

    def forward(self, x):
        # x = x.view(-1,28*28)
        output = self.fcnn(x)
        return output  

class FCNN(nn.Module):
    def __init__(self,input_shape, cls_num):
        super(FCNN, self).__init__()
        self.fcnn = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256,cls_num)
        )

    def forward(self, x):
        # x = x.view(-1,28*28)
        output = self.fcnn(x)
        return output  

class basic_NN(nn.Module):
    def __init__(self, input_shape, cls_num):
        super(basic_NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( input_shape , 1024),
            nn.ReLU(),
            nn.Linear( 1024 , 256),
            nn.ReLU(),
            nn.Linear( 256 , 128),
            nn.ReLU(),
            # nn.Linear( 256 , 128),
            nn.Linear( 128 , cls_num),
            # nn.Softmax()  ## don't use it, loss won't decrease
        )

    def forward(self, x):
        x = x.view(-1,28*28) # #for mnist
        pred = self.model(x)
        return pred



class basic_NN_dropout(nn.Module):
    def __init__(self, input_shape, cls_num):
        super(basic_NN_dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( input_shape , 1024),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear( 1024 , 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear( 256 , 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            # nn.Linear( 256 , 128),
            nn.Linear( 128 , cls_num),
            # nn.Softmax()  ## don't use it, loss won't decrease
        )

    def forward(self, x):
        x = x.view(-1,28*28) # #for mnist
        pred = self.model(x)
        return pred


class basic_CNN(nn.Module):
    def __init__(self):
        super(basic_CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 48, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            # nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.conv3 = nn.Sequential(         # input shape (48, 14, 14)
            nn.Conv2d(48, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 512)   # fully connected layer, output 10 classes
        self.out = nn.Linear( 512, 10)   # fully connected layer, output 10 classes
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        output = self.out(x)
        output = self.softmax(output)       # do softmax
        return output 

    

class Attacker_MLP(nn.Module):
    def __init__(self, n_input=10):
        super(Attacker_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_input,32),
            nn.ReLU(),
            nn.Linear(32,16),
            # nn.ReLU(),
            # nn.Linear(64,128),
            # nn.ReLU(),
            # nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(16,2),
            # nn.Linear(3,2),
        )

    def forward(self, x):
        output = self.mlp(x)
        return output    # return x for visualization



class RabdomForest:

    def __init__(self):
        self.rf = RandomForestClassifier()
        
    def train(self, data_x, data_y):
        self.rf.fit(data_x, data_y)
        print("finish training!")

    def predict(self, x):
        return self.rf.predict_proba(x)




class LogisticRegression(nn.Module):
    def __init__(self, input_shape, cls_num):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Sequential(
            nn.Linear(input_shape, cls_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.lr(x)
        return pred 



## for CIFAR-100 exp
class AlexNet(nn.Module):

  def __init__(self, classes=100):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
    #   nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
    #   nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


class AlexNet_dropout(nn.Module):

  def __init__(self, classes=100):
    super(AlexNet_dropout, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])