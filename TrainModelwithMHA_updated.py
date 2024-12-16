import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the dataset
solar_data = pd.read_csv('./data/k_final_solar_dataset_used.csv')

# Check for null values and normalize the data
scaler = MinMaxScaler()
solar_data_norm = pd.DataFrame(scaler.fit_transform(solar_data), columns=solar_data.columns)
print('solar final data', solar_data_norm)

# Separate features and target
X = solar_data_norm.drop('generation(Wh)', axis=1)
Y = solar_data_norm['generation(Wh)']

# Train-test-validation split: 80% train, 10% validation, 10% test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the RNN model with Multi-head Attention
class RNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=4):
        super(RNNWithAttention, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)  # Forward propagate the RNN

        # Attention mechanism
        out = out.permute(1, 0, 2)  # MultiHeadAttention expects (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(out, out, out)
        attn_output = attn_output.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_size)

        out = self.fc(attn_output[:, -1, :])  # Final output layer, take last sequence output
        return out

# Hyperparameters
input_size = 7
hidden_size = 64
output_size = 1
num_heads = 4
num_epochs = 200
learning_rate = 0.001

model_path = './trained_rnn_model_attention_b_7f.pth'

# Training Function
def train_model():
    model = RNNWithAttention(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_heads=num_heads)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor.unsqueeze(1))
        loss = criterion(outputs, Y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.unsqueeze(1))
            val_loss = criterion(val_outputs, Y_val_tensor)
            val_losses.append(val_loss.item())

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Load Model Function
def load_model():
    model = RNNWithAttention(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_heads=num_heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == "__main__":
    # Check if the model is already saved
    try:
        model = load_model()
        print("Model loaded from saved file.")
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        model = train_model()

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.unsqueeze(1)).squeeze().numpy()
        Y_test_actual = Y_test_tensor.squeeze().numpy()
        test_mse = mean_squared_error(Y_test_actual, test_predictions)
        test_rmse = np.sqrt(test_mse)

        print(f'Test MSE: {test_mse:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_actual, label='Actual Values', color='blue', marker='o')
    plt.plot(test_predictions, label='Predicted Values', color='red', linestyle='--', marker='x')
    plt.xlabel('Time (index)')
    plt.ylabel('Generation (Wh)')
    plt.title('Predicted vs Actual AC Power over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 1-hour prediction
    X_test_1hour = X_test_tensor[:60, :].unsqueeze(1)
    Y_test_1hour = Y_test_tensor[:60]
    with torch.no_grad():
        test_prediction_1hour = model(X_test_1hour).squeeze().numpy()
        Y_test_1hour_actual = Y_test_1hour.squeeze().numpy()

    minutes = np.arange(0, 60)
    plt.figure(figsize=(10, 6))
    plt.plot(minutes, Y_test_1hour_actual, label='Actual Generation', color='blue', marker='o')
    plt.plot(minutes, test_prediction_1hour, label='Predicted Generation', color='red', linestyle='--', marker='x')
    plt.xlabel('Minutes')
    plt.ylabel('Generation (Wh)')
    plt.title('Predicted vs Actual Generation for 1 Hour')
    plt.legend()
    plt.grid(True)
    plt.show()
