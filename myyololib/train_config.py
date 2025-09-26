from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.0005
    momentum: float = 0.937
    step_size: int = 1
    gamma: float = 0.25
    validation_frequency: int = 1
    print_info: bool = True

@dataclass
class FinetuneConfig:
    batch_size: int = 16
    num_epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0005
    momentum: float = 0.937
    step_size: int = 2
    gamma: float = 0.25
    validation_frequency: int = 1
    print_info: bool = True
