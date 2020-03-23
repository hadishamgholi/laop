import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import config as con
from models import Net
from time import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=con.batch_size,
                                          shuffle=True, num_workers=con.num_worker)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=con.batch_size,
                                         shuffle=False, num_workers=con.num_worker)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if con.use_cuda:
    net = Net().cuda()
else:
    net = Net()
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
print('num of net parameters:', sum(p.numel() for p in net.parameters()))


def train():
    epoch_train_loss = 0
    for batch in iter(trainloader):
        optimizer.zero_grad()
        inputs, labels = batch
        if con.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        out = net(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    return epoch_train_loss

def evaluate():
    epoch_test_loss = 0
    for batch in iter(testloader):
        inputs, labels = batch
        if con.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            out = net(inputs)
            loss = criterion(out, labels)
        epoch_test_loss += loss.item()
    return epoch_test_loss


def run():      
    for e in range(con.epochs):
        t1 = time()
        epoch_train_loss = train()
        epoch_test_loss = evaluate()
        t = time() - t1
        print(f"epoch {e}, train loss: {epoch_train_loss}, test loss: {epoch_test_loss}, in {t} secs")


if __name__ == "__main__":
    run()