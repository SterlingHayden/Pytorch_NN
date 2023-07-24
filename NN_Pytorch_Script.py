# %% [markdown]
# Pytorch Implementation

# %%
import numpy as np
import pickle 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.functional as F

# %% [markdown]
# ### Working with data

# %%
data_dict = pickle.load(open("cifar-2class-py2/cifar_2class_py2.p", "rb"), encoding="bytes")
# to call specifics of the dict call data_dict["name"]
train_data = data_dict[b'train_data']
train_labels = data_dict[b'train_labels']
test_data = data_dict[b'test_data']
test_labels = data_dict[b'test_labels']
m, n = train_data.shape
print(m, n)  # n = 24 x 24 pixels X 3 for RGB


# %%
# # Convert data and labels to torch tensors
# train_data = torch.from_numpy(train_data).float()
# #train_labels = torch.from_numpy(train_labels[:, 0]).long()  # Use only the first column as labels
# train_labels = torch.from_numpy(train_labels).long()
# test_data = torch.from_numpy(test_data).float()
# test_labels = torch.from_numpy(test_labels).long()

# %%

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# Convert data and labels to torch tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels[:, 0]).long()  # Use only the first column as labels
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels[:, 0]).long()  # Use only the first column as labels

# Create train and test datasets
train = CustomDataset(train_data, train_labels)
test = CustomDataset(test_data, test_labels)

# %%
#train = np.concatenate((train_data, train_labels), 1)
#test = np.concatenate((test_data, test_labels), 1)
# train = torch.cat((train_data, train_labels), dim=1)
# test = torch.cat((test_data, test_labels), dim=1)

# %%
batch_size = 10

# Create data loaders.
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size)

# train_dataset = CIFAR2ClassDataset(train_data, train_labels)
# test_dataset = CIFAR2ClassDataset(test_data, test_labels)

next(iter(train_dataloader))

# %% [markdown]
# ### Creating model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n,100),
            nn.ReLU(),
            nn.Linear(100,2)#,
            #nn.Softmax()
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        prob = F.softmax(logits, dim=1)
        return prob

model = NeuralNetwork().to(device)
print(model)

# %% [markdown]
# ### Optimizing the Model Parameters

# %%
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=1e-4)

# %%
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, data in enumerate(dataloader):
#         X, y = data[:, :-1], data[:, -1]
#         # X = X.float().to(device)
#         # y = y.long().to(device)  
#         #X, y = X.to(device), y.to(device)
        
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
        
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss_value = loss.item()
#             current = batch * len(X)
#             print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


# %%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        # X, y = data[:, :-1], data[:, -1]
        # X = X.float().to(device)
        # y = y.long().to(device)  
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


# %%
# def test(dataloader, model):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for data in dataloader:
#             X, y = data[:, :-1], data[:, -1]
#             # X = X.float().to(device)
#             # y = y.long().to(device)
#             #X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= size
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X,y) in dataloader:
            #X, y = data[:, :-1], data[:, -1]
            # X = X.float().to(device)
            # y = y.long().to(device)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")


