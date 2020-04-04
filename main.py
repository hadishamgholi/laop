import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import config as con
from models import Net
from time import time
from utils import accuracy, get_cifar10_data, set_random_seeds, get_model_layers, get_prms_rqr_grd
try:
    from apex import amp
except:
    print('the apex module does not exists')

set_random_seeds(con.random_seed)
trainset, trainloader, testset, testloader = next(get_cifar10_data())


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
    for i, batch in enumerate(iter(trainloader)):
        print(f'\riter {i}/{len(trainloader)}', end='')
        inputs, labels = batch
        if con.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
            
        lyrs, lyrs_cnt = get_model_layers(net, True)
        for l in reversed(lyrs):
            for ll in lyrs:
                ll.requires_grad_(False)
            l.requires_grad_(True)
            optimizer.param_groups = []
            optimizer.add_param_group(
                {'params' : get_prms_rqr_grd(net)}
            )
            out = net(inputs)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            if con.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / lyrs_cnt
            epoch_train_acc += accuracy(out, labels) / lyrs_cnt
    print()
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
    pass