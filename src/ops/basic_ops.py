import torch


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        supported_consensus_types = ["avg", "max", "identity"]
        super().__init__()
        self.consensus_type = consensus_type.lower()
        if self.consensus_type not in supported_consensus_types:
            raise ValueError(
                f"Unknown consensus type '{consensus_type}', expected one of {supported_consensus_types}"
            )
        self.dim = dim

    def forward(self, input):
        if self.consensus_type == "avg":
            return input.mean(dim=self.dim)
        elif self.consensus_type == "max":
            return input.max(dim=self.dim)
        elif self.consensus_type == "identity":
            return input
        else:
            raise ValueError(f"Unknown consensus_type '{self.consensus_type}'")
