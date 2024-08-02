import torch.nn as nn

class base(nn.Module):
    def __init__(self,in_ch,Stride=1,n=1):
        super(base,self).__init__()
        self.con1=nn.Conv2d(in_channels=in_ch,out_channels=64*n,kernel_size=1,stride=Stride)
        self.bb1=nn.BatchNorm2d(64*n)
        self.con2=nn.Conv2d(in_channels=64*n,out_channels=64*n,kernel_size=5,stride=1,padding=2)
        self.bb2=nn.BatchNorm2d(64*n)
        self.con3=nn.Conv2d(in_channels=64*n,out_channels=256*n,kernel_size=1,stride=1)
        self.bb3=nn.BatchNorm2d(256*n)
        self.relu=nn.LeakyReLU(0.03)
        self.connect=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=256*n,kernel_size=1,stride=Stride),nn.BatchNorm2d(256*n))

    def forward(self,x):
        out=self.bb3(self.con3(self.relu(self.bb2(self.con2(self.relu(self.bb1(self.con1(x))))))))
        out+=self.connect(x)
        return self.relu(out)
    

class resnet_50(nn.Module):
    def __init__(self,classes):
        super(resnet_50,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #layer1
            base(in_ch=64),
            base(in_ch=256),
            base(in_ch=256),
            #layer2
            base(in_ch=256,Stride=2,n=2),
            base(in_ch=512,Stride=1,n=2),
            base(in_ch=512,Stride=1,n=2),
            base(in_ch=512,Stride=1,n=2),
            #layer3
            base(in_ch=512,Stride=2,n=3),
            base(in_ch=1024,Stride=1,n=3),
            base(in_ch=1024,Stride=1,n=3),
            base(in_ch=1024,Stride=1,n=3),
            base(in_ch=1024,Stride=1,n=3),
            base(in_ch=1024,Stride=1,n=3),
            #layer 
            base(in_ch=1024,Stride=2,n=2),
            base(in_ch=2048,Stride=1,n=2),
            base(in_ch=2048,Stride=1,n=2),

            nn.Flatten(),
            nn.Linear(in_features=2048*7*7,out_features=1024),
            nn.LeakyReLU(0.03),
            nn.Linear(1024,classes)
        )

    def forward(self,x):
        return self.model(x)






        