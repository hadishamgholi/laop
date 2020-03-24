import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import config as con
from models import Net
from time import time
from utils import accuracy
try:
    from apex import amp
except:
    print('the apex module does not exists')

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
if con.use_apex:
    net, optimizer = amp.initialize(net, optimizer, opt_level=con.apex_opt_level)
criterion = nn.CrossEntropyLoss()
print('num of net parameters:', sum(p.numel() for p in net.parameters()))


def train():
    epoch_train_loss = 0
    epoch_train_acc = 0
    for batch in iter(trainloader):
        optimizer.zero_grad()
        inputs, labels = batch
        if con.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        out = net(inputs)
        loss = criterion(out, labels)
        if con.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy(out, labels)
    return epoch_train_loss, epoch_train_acc / len(trainset)


def evaluate():
    epoch_test_loss = 0
    epoch_test_acc = 0
    for batch in iter(testloader):
        inputs, labels = batch
        if con.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            out = net(inputs)
            loss = criterion(out, labels)
        epoch_test_loss += loss.item()
        epoch_test_acc += accuracy(out.data, labels)
    return epoch_test_loss, epoch_test_acc / len(testset)


def run():      
    for e in range(con.epochs):
        t1 = time()
        tr_loss, tr_acc = train()
        te_loss, te_acc = evaluate()
        t = time() - t1
        print("epoch {}| train loss: {:.5f}, acc: {:.5f} | test loss: {:.5f}, acc: {:.5f} | in {:.5f} secs".format(
            e, tr_loss, tr_acc, te_loss, te_acc, t
        ))


if __name__ == "__main__":
    run()