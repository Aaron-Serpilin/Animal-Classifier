import torch
from torch import nn

class CustomModel(nn.Module):

    """
    Creates the Custom CNN Architecture.
    """

    def __init__(
            self,
            input_shape: int,
            hidden_units: int,
            output_shape: int
        ) -> None:
        super().__init__()
        # We can update these layers later with our Custom Architecture
        self.conv_block_1 = nn.Sequential()
        self.conv_block_2 = nn.Sequential()
        self.classifier = nn.Sequential()

    def forward(self, x: torch.Tensor):
        # We do it in one line due to operator fusion
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
    
