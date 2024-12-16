import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


# Define the Seq2Seq Model with Attention
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=4, dropout=0.2):
        super(Seq2SeqWithAttention, self).__init__()

        # Encoder: RNN to process the input sequence
        self.encoder_rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder: GRU with attention mechanism
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

        # Fully connected layers for output
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Encoder
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        encoder_out, encoder_hidden = self.encoder_rnn(x, h0)

        # Decoder with Attention
        decoder_input = encoder_out  # Using the encoder output as the decoder input
        decoder_out, _ = self.decoder_gru(decoder_input)

        # Attention mechanism
        decoder_out = decoder_out.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(decoder_out, decoder_out, decoder_out)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)

        # Output layer (last time step)
        out = self.fc1(attn_output[:, -1, :])  # (batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, output_size)

        return out


# Define hyperparameters
input_size = 11  # Number of features (input features)
hidden_size = 64  # Hidden layer size
output_size = 1  # Output is generation in Wh (regression task)
num_heads = 4  # Number of attention heads
learning_rate = 0.001
num_epochs = 200


# Training function
def train_model(model, criterion, optimizer, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor,
                num_epochs=200):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, Y_val_tensor)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses


# Evaluation function
def evaluate_model(model, X_test_tensor, Y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_predictions = test_predictions.squeeze().numpy()
        Y_test_actual = Y_test_tensor.squeeze().numpy()
        test_mse = mean_squared_error(Y_test_tensor, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(Y_test_actual, test_predictions)
    return test_mse, test_rmse, test_r2, test_predictions, Y_test_actual


# Plotting function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Seq2Seq prediction (predicting the entire sequence at once)
def seq2seq_predict(model, X_input, timesteps):
    # Predict the entire sequence in one go
    with torch.no_grad():
        predicted_output = model(X_input)
    return predicted_output.squeeze().numpy()


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import torch.optim as optim

    # Load and normalize data
    solar_data = pd.read_csv('./data/k_final_dataset_withseason.csv')
    scaler = MinMaxScaler()
    solar_data_norm = pd.DataFrame(scaler.fit_transform(solar_data), columns=solar_data.columns)

    # Split features and target
    X = solar_data_norm.drop('generation(Wh)', axis=1)
    Y = solar_data_norm['generation(Wh)']

    # Train-test split
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)

    # Instantiate model, loss function, and optimizer
    model = Seq2SeqWithAttention(input_size, hidden_size, output_size, num_heads=num_heads)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_losses, val_losses = train_model(model, criterion, optimizer, X_train_tensor, Y_train_tensor, X_val_tensor,
                                           Y_val_tensor, num_epochs)
    plot_losses(train_losses, val_losses)

    # Evaluate on test set
    test_mse, test_rmse, test_r2, test_predictions, Y_test_actual = evaluate_model(model, X_test_tensor, Y_test_tensor)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test R² Score: {test_r2:.4f}')

    # Seq2Seq: Predict for the next 60 time steps (entire sequence)
    X_test_input = X_test_tensor[:1, :]  # First sample from the test set
    predictions = seq2seq_predict(model, X_test_input, 60)

    # Plot predictions vs actual for 60 steps
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, 'rx-', label='Predicted')
    plt.plot(Y_test_tensor[:60].numpy(), 'bo-', label='Actual')
    plt.title('Seq2Seq 60-Days Predictions vs Actual')
    plt.xlabel('Days')
    plt.ylabel('Generation (Wh)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate R² score for the 60-day predictions
    Y_test_actual_60 = Y_test_tensor[:60].numpy()
    test_r2_60 = r2_score(Y_test_actual_60, predictions)
    print(f'R² Score for the next 60 days: {test_r2_60:.4f}')
