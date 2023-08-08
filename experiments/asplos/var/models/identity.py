import torch


class Identity(torch.nn.Module):
    def __init__(self, block_id: int, name: str):
        super().__init__()
        self.block_id = block_id
        self.name = name

    def forward(self, x):
        return x
