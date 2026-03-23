# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

## train set:


<img width="761" height="830" alt="image" src="https://github.com/user-attachments/assets/1667d7f9-a824-4f15-80e9-c694729b3941" />



## test set:


<img width="726" height="821" alt="image" src="https://github.com/user-attachments/assets/0df77723-53f7-477a-9913-73ab90a2f1bd" />


## Design Steps

### Step 1:
Import necessary libraries.

### Step 2:
Load and preprocess the data.

### Step 3:
Create input-output sequences.

### Step 4:
Convert data to PyTorch tensors.

### Step 5:
Define the RNN model.

### Step 6:
Train the model using the training data.

### Step 7:
Evaluate the model and plot predictions.


## Program
#### Name: S.Navadeep
#### Register Number: 212224230180

```Python 


# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")




```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="801" height="574" alt="image" src="https://github.com/user-attachments/assets/200f8c26-1b37-42de-81dd-a738b1b5de7b" />

### Predictions 

<img width="980" height="658" alt="image" src="https://github.com/user-attachments/assets/fd9146c0-eaf4-46b4-92af-3edeb4179203" />

<img width="315" height="60" alt="image" src="https://github.com/user-attachments/assets/7a9f856c-afca-4a54-8c8d-18c4a2874f7a" />

## Result

The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.


