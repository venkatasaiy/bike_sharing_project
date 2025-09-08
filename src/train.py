import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_and_preprocess_data
from model import BikeSharingModel
import pickle

def train_model():
    """
    Train the bike-sharing model and save it.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("data/bike_sharing.csv")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = BikeSharingModel(input_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Save the model and input size
    torch.save({
        'input_size': input_size,
        'state_dict': model.state_dict()
    }, "models/bike_sharing.pkl")
    
    # Save the preprocessor
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    print("Model and preprocessor saved successfully.")

if __name__ == "__main__":
    train_model()