import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import IrisClassificationNetwork

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 42

#Prepare the data (load, transform, split)
iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Convert to torch.tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#Instanciate the model
model = IrisClassificationNetwork()

#Define loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training loop
epochs = 100
batch_size = 16

for epoch in tqdm(range(epochs)):
    model.train()

    #Permute the data and batch it
    permutation = torch.randperm(X_train.size(0))

    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train[indices]
        batch_y = y_train[indices]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Calc loss
        total_loss += loss.item()
    
    #Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).float().mean()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {accuracy:.4f}")

torch.save(model.state_dict(), "iris-model.pth")