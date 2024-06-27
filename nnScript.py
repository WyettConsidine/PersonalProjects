import torch 
from torch import nn
from torch.nn import functional as F
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt


#This Function accesses the online pytorch databases to retrieve the FashionMNIST dataset. If it is already downloaded, it is 
#stored localy.
def getData():
    #Import the training data
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        #NOTE: the following target_transformation lambda one-hot encodes the lables
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
    #Import the testing data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        #NOTE: the following target_transformation lambda one-hot encodes the lables
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
    #return both datasets.
    return training_data, test_data

def displData():
    #use the get data funciton to import the train/test data
    training_data, testing_data = getData()

    #define label map
    labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
    }

    #Define plt figures frame
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    #Itterate 9 times:
    for i in range(1, cols * rows + 1):
        #Retrieve a sample tensor index
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        #define its img tensor and label
        img, label = training_data[sample_idx]
        #add the figure to the subplot
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        #Display the img tensor on the desingated plot
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def makeDataLoader(training_data,testing_data):
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
    return train_dataloader,test_dataloader



###Define NN class. Basic NN will have an init sequential stack of layers.
### Layers are as follows: 28*28 mat -> 512, ReLU activ. Func,
#  512 -> 512, ReLU Activ. Func, 512 -> 10

class NeuralNetwork(nn.Module):
    #define an init fuciton for the class
    def __init__(self):
        super().__init__()
        #define flatten fuction
        self.flatten = nn.Flatten()
        #define the sequantial stack
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    #define the feedforward applicaiton funciton
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=10, kernel_size=(3,3), stride = (1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), stride = (1,1), padding=(1,1))
        # self.fc1 = nn.Linear(16*7*7,216)
        # self.fc2 = nn.Linear(216,num_classes)
        self.fc1 = nn.Linear(20*7*7,num_classes)

    def forward(self,input):
        X = F.relu(self.conv1(input))
        X = self.pool(X)
        X = F.relu(self.conv2(X)) 
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.fc1(X)
        #X = self.fc2(X)
        
        return X


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn): 
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y = torch.tensor([(yEntry == 1).nonzero(as_tuple=True)[0] for yEntry in y])
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct/= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def trainNN( NNFunc, save_model = False, Model_Name = 'model.pth'):
    print("Executing")
    #getData()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NNFunc().to(device)
    print(model)
    train_D, test_D = getData()
    train_DL, test_DL = makeDataLoader(train_D,test_D)

    ##HyperParameters:
    epochs = 5
    batch_size = 64
    learning_rate = 5e-3

    ###Loss Funciton:
    loss_fn = nn.CrossEntropyLoss()

    ### Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_DL, model, loss_fn, optimizer, batch_size)
        test_loop(test_DL, model, loss_fn)

    if save_model == True:
        torch.save(model, Model_Name)

    print("Done!")

def loadModel(pth_path):
    model = torch.load(pth_path)
    return model

def dispModelFunc(modelName):
    model = loadModel(modelName)
    _, testData = getData()
    labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
    }
    #Define plt figures frame
    figure = plt.figure(figsize=(13, 10))
    cols, rows = 3,3
    #Itterate 9 times:
    _, testing_data = getData()
    test_dataloader = iter(DataLoader(testing_data, batch_size=1, shuffle=True))
    for i in range(1, cols * rows + 1):
        with torch.no_grad():
            X,y = next(test_dataloader)
            pred = model(X).argmax(1)
            label = y.argmax(1)
        #add the figure to the subplot
        figure.add_subplot(rows, cols, i)
        plt.title(f"True: {labels_map[label.item()]}, Predicted: {labels_map[pred.item()]}")
        plt.axis("off")
        #Display the img tensor on the desingated plot
        plt.imshow(X.squeeze(), cmap="gray")
    plt.show()

if __name__ == "__main__":
    #trainNN(NNFunc = ConvolutionalNeuralNetwork, save_model = True, Model_Name = 'CNN.pth')
    #model = loadModel('5EpochCNN.pth')
    dispModelFunc('CNN.pth')