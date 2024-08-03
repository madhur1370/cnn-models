from resnet50 import resnet_50
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms

transform=transforms.Compose({
    transforms.Resize((224,224)),
    transforms.ToTensor()
})


device="cuda:0" if torch.cuda.is_available() else "cpu"
train,test=ImageFolder(root='dataset/Training',transform=transform),ImageFolder(root='dataset/Testing',transform=transform)

classes=train.classes
model=resnet_50(classes=len(classes)).to(device=device)

train_loader,test_loader=DataLoader(dataset=train,shuffle=True,batch_size=4),DataLoader(dataset=test,shuffle=True,batch_size=4)

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=2e-3,weight_decay=2e-5)

epochs = 500
for epoch in range(epochs):
    model.train()
    train_loss=0
    for image,target in train_loader:
        image,target=image.detach().to(device=device),target.detach().to(device=device)
        output=model(image.float()/255.0)
        loss=loss_fn(input=output,target=target)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    cor=0
    with torch.no_grad():
        for image,target in test_loader:
            image,target=image.detach().to(device=device),target.detach().to(device=device)
            output=model(image)
            loss=loss_fn(output,target)
            _,predicted=torch.max(output,dim=1)
            cor+=(predicted==target).sum().item()
    
    print(f'{epoch+1}/{epochs}  train loss = {train_loss} accuracy {cor/200}')

    
torch.save({
    'model-state':model.state_dict(),
    'optim-state':optimizer.state_dict(),
    'classes':classes
},'vehicle.pth')

