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

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_sizes, activation='', use_batchnorm=[False], dropout_prob=[0.0]):
        super(LinearModel, self).__init__()
        self.input_size = input_size  # Set input_size as an attribute
        layers = []
        in_size = input_size
        for i in range(num_layers):
            out_size = hidden_sizes[i]
            layers.append(nn.Linear(in_size, out_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if use_batchnorm[i]:
                layers.append(nn.BatchNorm1d(out_size))
            if dropout_prob[i] > 0:
                layers.append(nn.Dropout(dropout_prob[i]))
            in_size = out_size
      
        layers.append(nn.Linear(in_size, output_size))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        return nn.Softmax(dim=1)(x)

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

activation = ''
use_batchnorm = False
dropout = 0.0
num_layers = 1
hidden_sizesb = 100
hidden_sizes = [hidden_sizesb, hidden_sizesb, hidden_sizesb]
n_epo = 5

optimiz = ['SGD', 'RMSProp', 'Adam']
lern_rate = np.logspace(-3, -1, 3)

for optim in optimiz:
    for lr in lern_rate:
        model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_size = hidden_size, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
        if optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = lr) 
        elif optim == 'RMSProp':
            optimizer = torch.optim.RMSprop(model.parameters(), lr = lr) 
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
        criterion = torch.nn.CrossEntropyLoss().to(torch_device)
        start_time = time.time()
        train(model, torch_device, train_loader, optimizer, criterion, n_epo)
        print("Training time:", time.time()-start_time)
        print("Train Accuracy:", test(model, torch_device, train_loader))
        print("Test Accuracy:", test(model, torch_device, test_loader))

hidden_sizes = [100, 100, 25, 25]
n_epochs = [5,10,15,20,25]

for num_layers in range (1,3):
    for n_epo in n_epochs:
        model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
        criterion = torch.nn.CrossEntropyLoss().to(torch_device)
        start_time = time.time()
        train(model, torch_device, train_loader, optimizer, criterion, n_epo)
        print("Training time:", time.time()-start_time)
        print("Train Accuracy:", test(model, torch_device, train_loader))
        print("Test Accuracy:", test(model, torch_device, test_loader))

n_epo = 5
num_layers = 1
hidden_sizesb = 25
hidden_sizes = [hidden_sizesb, hidden_sizesb, hidden_sizesb]

for i in range (10):
    model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
    criterion = torch.nn.CrossEntropyLoss().to(torch_device)
    start_time = time.time()
    train(model, torch_device, train_loader, optimizer, criterion, n_epo)
    print("Training time:", time.time()-start_time)
    print("Train Accuracy:", test(model, torch_device, train_loader))
    print("Test Accuracy:", test(model, torch_device, test_loader))

n_epo = 10
hidden_sizesL = [10, 25, 50, 100, 200, 250, 500]

for num_layers in range (1,3):
    for hidden_sizesLB in hidden_sizesL:
          hidden_sizes = [hidden_sizesLB, hidden_sizesLB, hidden_sizesLB, hidden_sizesLB]
          model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
          criterion = torch.nn.CrossEntropyLoss().to(torch_device)
          start_time = time.time()
          train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
          print("Training time:", time.time()-start_time)
          print("Train Accuracy:", test(model, torch_device, train_loader))
          print("Test Accuracy:", test(model, torch_device, test_loader))

hidden_sizesb = 25
hidden_sizes = [hidden_sizesb, hidden_sizesb, hidden_sizesb]
activationL = ['',’relu’]

for num_layers in range (1,4):
    for activation in activationL:
        model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(torch_device)
        start_time = time.time()
        train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
        print("Training time:", time.time()-start_time)
        print("Train Accuracy:", test(model, torch_device, train_loader))
        print("Test Accuracy:", test(model, torch_device, test_loader))

use_batchnormL = [False, True]
num_layers = 1

for activation in activationL:
    for use_batchnorm in use_batchnormL:
        model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(torch_device)
        start_time = time.time()
        train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
        print("Training time:", time.time()-start_time)
        print("Train Accuracy:", test(model, torch_device, train_loader))
        print("Test Accuracy:", test(model, torch_device, test_loader))

num_layers = 2
dropout_prob = [0.0, 0.0]

for activation in activationL:
    for use_batchnorm1 in use_batchnormL:
        for use_batchnorm2 in use_batchnormL:
            use_batchnorm = [use_batchnorm1, use_batchnorm2]
            model = model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model, torch_device, train_loader))
            print("Test Accuracy:", test(model, torch_device, test_loader))

use_batchnorm = False
dropout_probL = [0.0, 0.1]
num_layers = 1

for activation in activationL:
    for dropout_prob in dropout_probL:
        model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(torch_device)
        start_time = time.time()
        train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
        print("Training time:", time.time()-start_time)
        print("Train Accuracy:", test(model, torch_device, train_loader))
        print("Test Accuracy:", test(model, torch_device, test_loader))

num_layers = 2
use_batchnorm = [False, False]

for activation in activationL:
    for dropout_prob1 in dropout_probL:
        for dropout_prob2 in dropout_probL:
            dropout_prob = [dropout_prob1, dropout_prob2]
            model = model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model, torch_device, train_loader))
            print("Test Accuracy:", test(model, torch_device, test_loader))

num_layers = 1
activationL = ['',’relu’]
use_batchnormL = [False, True]
dropout_probL = [0.0, 0.1]

for activation in activationL:
    for use_batchnorm in use_batchnormL:
        for dropout_prob in dropout_probL:
            model = model = LinearModel(input_size = 28*28, output_size = 10, num_layers = num_layers, hidden_sizes = hidden_sizes, activation = activation, use_batchnorm = use_batchnorm, dropout_prob = dropout_prob)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss().to(torch_device)
            start_time = time.time()
            train(model, torch_device, train_loader, optimizer, criterion, n_epochs)
            print("Training time:", time.time()-start_time)
            print("Train Accuracy:", test(model, torch_device, train_loader))
            print("Test Accuracy:", test(model, torch_device, test_loader))
