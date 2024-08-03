import torch.nn as nn
class vgg19(nn.Module):
    def __init__(self,classes):
        super(vgg19,self).__init__()
        self.model=nn.Sequential(
                                 nn.Conv2d(3,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64,64,3,1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2),

                                 nn.Conv2d(64,128,3,1,1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128,128,3,1,1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2),

                                 nn.Conv2d(128,256,3,1,1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.Conv2d(256,256,3,1,1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.Conv2d(256,256,3,1,1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2),

                                 nn.Conv2d(256,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2),

                                 nn.Conv2d(512,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512,512,3,1,1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2),

                                 nn.Flatten(),
                                 nn.Linear(7*7*512,4096),
                                 nn.ReLU(),
                                 nn.Linear(4096,classes)
                    )
        
    def forward(self,x):
        return self.model(x)
        