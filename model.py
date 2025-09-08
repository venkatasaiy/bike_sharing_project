import torch
import torch.nn as nn
class BikeSharingModel:
    # Your model implementation here
    pass
class BikeSharingModel(nn.Module):
    """
    Neural network model for bike-sharing demand prediction.
    
    Parameters:
    - input_size (int): Number of input features.
    """
    def __init__(self, input_size):
        super(BikeSharingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output is a single value (count)
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Parameters:
        - x (Tensor): Input tensor.
        
        Returns:
        - Tensor: Predicted bike-sharing demand.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Test the model
    model = BikeSharingModel(input_size=10)  # Example input size
    sample_input = torch.randn(1, 10)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")