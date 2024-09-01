import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets as dts
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time

# Load MNIST dataset
traindt = dts.MNIST(root='data', train=True, transform=ToTensor(), download=True)
testdt = dts.MNIST(root='data', train=False, transform=ToTensor())

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(traindt, batch_size=32, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(testdt, batch_size=32, shuffle=False, num_workers=1)

class CNN_mnist(nn.Module):
    def __init__(self, output_size,
                 num_layersC, num_layersL,
                 hidden_sizesC, hidden_sizesL,
                 activationC='', activationL='',
                 use_batchnormC=[False], use_batchnormL=[False],
                 dropout_probC=[0.0], dropout_probL=[0.0],
                 kernel_size=3, stride=1, padding=1, dilation=1):
        super(CNN_mnist, self).__init__()
        height, width = 28, 28

        layersC = []
        in_size = 1
        for i in range(num_layersC-1):
            out_size = hidden_sizesC[i]
            layersC.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
            if activationC == 'relu':
                layersC.append(nn.ReLU())
            elif activationC == 'sigmoid':
                layersC.append(nn.Sigmoid())
            elif activationC == 'tanh':
                layersC.append(nn.Tanh())
            if use_batchnormC[i]:
                layersC.append(nn.BatchNorm2d(out_size))
            if dropout_probC[i] > 0:
                layersC.append(nn.Dropout2d(dropout_probC[i]))
            in_size = out_size
            height = (height - kernel_size * dilation + 2 * padding) // stride + 1
            width = (width - kernel_size * dilation + 2 * padding) // stride + 1

        layersC.append(nn.Conv2d(in_size, hidden_sizesC[num_layersC-1], kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv = nn.Sequential(*layersC)

        layersL = []
        layersL.append(nn.Flatten())
        in_size = hidden_sizesC[-1] * height * width
        for i in range(num_layersL):
            out_size = hidden_sizesL[i]
            layersL.append(nn.Linear(in_size, out_size))
            if activationL == 'relu':
                layersL.append(nn.ReLU())
            elif activationL == 'sigmoid':
                layersL.append(nn.Sigmoid())
            elif activationL == 'tanh':
                layersL.append(nn.Tanh())
            if use_batchnormL[i]:
                layersL.append(nn.BatchNorm1d(out_size))
            if dropout_probL[i] > 0:
                layersL.append(nn.Dropout(dropout_probL[i]))
            in_size = out_size
        layersL.append(nn.Linear(in_size, output_size))
        self.lin = nn.Sequential(*layersL)

    def forward(self, x):
        after_conv = self.conv(x)
        output     = self.lin(after_conv)
        return nn.LogSoftmax(dim=1)(output)

def train(model, device, train_loader, optimizer, criterion, n_epochs):
    model.to(device)
    model.train()
    for epoch in tqdm(range(n_epochs)):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

def test(model, device, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

output_size = 10
kernel_size = 3
stride = 1
padding = 1
dilation = 1
n_epochs = 10
use_batchnormC = False
use_batchnormL = False
dropout_probC = 0.0
dropout_probL = 0.0
activationC = ''
activationL = ''

num_layersL = 1

hidden_sizesCLB = [10, 25, 50, 100, 200, 250, 500]
hidden_sizesL = [25, 25, 25, 25]
for num_layersC in range (2,4):
    for hidden_sizesCL in hidden_sizesCLB:
          hidden_sizesC = [hidden_sizesCL, hidden_sizesCL, hidden_sizesCL, hidden_sizesCL]
          model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
          criterion = torch.nn.CrossEntropyLoss().to(torch_device)
          start_time = time.time()
          train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
          print("Training time:", time.time()-start_time)
          print("Train Accuracy:", test(model, torch_device, train_loader))
          print("Test Accuracy:", test(model, torch_device, test_loader))

activationCL = ['',’relu’]
hidden_sizesL = [25, 25, 25, 25]
hidden_sizesL = [25, 25, 25, 25]

for num_layersC in range (2,4):    
    for num_layersL in range (1,3):
        for activationC in activationCL:
            for activationL in activationCL:
                model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.CrossEntropyLoss().to(torch_device)
                start_time = time.time()
                train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
                print("Training time:", time.time()-start_time)
                print("Train Accuracy:", test(model, torch_device, train_loader))
                print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersC = 2
num_layersL = 1
dropout_probC = 0.0
dropout_probL = 0.0

use_batchnormCL = [False, True]
for activationCLB in activationCL:
    activationC = activationCLB
    activationL = activationCLB
    for use_batchnormC in use_batchnormCL:
        for use_batchnormL in use_batchnormCL:
            model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model, torch_device, train_loader))
            print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersC = 3
num_layersL = 2 
dropout_probC = [0.0, 0.0]
dropout_probL = [0.0, 0.0]

for activationCLB in activationCL:
    activationC = activationCLB
    activationL = activationCLB
    for use_batchnormC1 in use_batchnormCL:
        for use_batchnormC2 in use_batchnormCL:
            for use_batchnormL1 in use_batchnormCL:
                for use_batchnormL2 in use_batchnormCL:
                    use_batchnormC = [use_batchnormC1, use_batchnormC2]
                    use_batchnormL = [use_batchnormL1, use_batchnormL2]
                    model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = torch.nn.CrossEntropyLoss().to(torch_device)
                    start_time = time.time()
                    train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
                    print("Training time:", time.time()-start_time)
                    print("Train Accuracy:", test(model, torch_device, train_loader))
                    print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersC = 2
num_layersL = 1
use_batchnormC = False
use_batchnormL = False

for activationCLB in activationCL:
    activationC = activationCLB
    activationL = activationCLB
    for dropout_probC in dropout_probCL:
        for dropout_probL in dropout_probCL:
            model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model, torch_device, train_loader))
            print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersC = 3
num_layersL = 2
use_batchnormC = [False, False]
use_batchnormL = [False, False]
     
for activationCLB in activationCL:
    activationC = activationCLB
    activationL = activationCLB
    for dropout_probC1 in dropout_probCL:
        for dropout_probC2 in dropout_probCL:
            for dropout_probL1 in dropout_probCL:
                for dropout_probL2 in dropout_probCL:
                    dropout_probC = [dropout_probC1, dropout_probC2]
                    dropout_probL = [dropout_probL1, dropout_probL2]
                    model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = torch.nn.CrossEntropyLoss().to(torch_device)
                    start_time = time.time()
                    train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
                    print("Training time:", time.time()-start_time)
                    print("Train Accuracy:", test(model, torch_device, train_loader))
                    print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersL = 2
num_layersL = 1
use_batchnormCL = [False,True]
dropout_probCL = [0.0,0.1]

for activationCL in activation:
    activationC = activationCL
    activationL = activationCL
    for use_batchnormC in use_batchnorm:
        for use_batchnormL in use_batchnorm:
            for dropout_probC in dropout_prob:
                for dropout_probL in dropout_prob:
                    model = CNN_mnist(output_size, num_layersC, num_layersL, hidden_sizesC, hidden_sizesL, activationC, activationL, use_batchnormC, use_batchnormL, dropout_probC, dropout_probL, kernel_size, stride, padding, dilation)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = torch.nn.CrossEntropyLoss().to(torch_device)
                    start_time = time.time()
                    train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
                    print("Training time:", time.time()-start_time)
                    print("Train Accuracy:", test(model, torch_device, train_loader))
                    print("Test Accuracy:", test(model, torch_device, test_loader))

num_layersL = 2
num_layersL = 1

use_batchnormC = True
use_batchnormL = False
dropout_probC = 0.0
dropout_probL = 0.0
activationC = 'relu'
activationL = 'relu'

kernel_sizew = [3,5]
stridew = [1,3]
paddingw = [1,3]
dilationw = [1,3]

for kernel_size in kernel_sizew:
    for stride in stridew:
        for padding in paddingw:
            for dilation in dilation:
        if kernel_size == kernel_sizew[1] and stride == stridew[1] and padding == paddingw[0] and dilation == dilation[1]:
            continue
        else:
            model1 = CNN_mnistWOS(output_size, hidden_sizesC, hidden_sizesL, kernel_size, stride, padding, dilation)
            optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model1, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model1, torch_device, train_loader))
            print("Test Accuracy:", test(model1, torch_device, test_loader))
