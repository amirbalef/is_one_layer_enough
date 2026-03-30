from dataclasses import dataclass, field
from typing import Optional
import ticla
import inspect
import os

# -----------------------
# General model Config
# -----------------------
@dataclass
class GeneralConfig:
    name: str = "nanotabpfn"
    variant: str = "looped_nanotabpfn"
    class_name : str = "LoopedNanoTabPFN"
    weights_path : str = os.path.abspath(os.path.dirname(inspect.getfile(ticla)) + "/..") + "/workdir/looped_nanotabpfn/checkpoint/step-950.ckpt"

# -----------------------
# Model Architecture Config
# -----------------------
@dataclass
class ModelConfig:
    embedding_size: int = 192
    num_attention_heads: int = 6
    num_layers: int = 6
    num_outputs: int = 10
    mlp_hidden_size:  int = 768

# -----------------------
# Training Config
# -----------------------
@dataclass
class TrainingConfig:
    amp: bool = True
    model_compile: bool = False
    device: str = "cuda"
    dtype: str = "float32"
    np_seed: int = 42
    torch_seed: int = 42
    max_steps: int = 100000
    batch_size: int = 512
    micro_batch_size: int = 4
    lr: float = 1e-4
    scheduler: str = "cosine_warmup"
    warmup_proportion: float = 0.02
    warmup_steps: int = 2000
    gradient_clipping: float = 1.0
    weight_decay: float = 0.0
    cosine_num_cycles: int = 1
    cosine_amplitude_decay: float = 1.0
    cosine_lr_end: float = 0.0
    poly_decay_lr_end: float = 1e-7
    poly_decay_power: float = 1.0
    freeze_layers: list = field(default_factory=lambda: []) #["col_embedder", "row_embedder", "icl_predictor"]

# -----------------------
# Prior Dataset Config
# -----------------------
@dataclass
class PriorConfig:
    prior_dir: str =  "./workdir/nanotabpfn/prior/dataset/"
    prior_type: str = "mix_scm"
    prior_device: str = "cpu"
    batch_size_per_gp: int = 4
    num_batches : int = 10000
    min_features: int = 2
    max_features: int = 30
    max_classes: int = 10
    min_seq_len: Optional[int] = None
    max_seq_len: int = 1024
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: float = 0.1
    max_train_size: float = 0.9
    replay_small: bool = False
    n_jobs: int = -1
    num_threads_per_generate : int = 1
    resume_from : int = 0
    delete_after_load:  bool = False
    load_prior_start: int = 0
    save_dir: str = "./workdir/nanotabpfn/prior/dataset/"
    
    


# -----------------------
# Checkpoint Config
# -----------------------
@dataclass
class CheckpointConfig:
    dir: Optional[str] = "./workdir/looped_nanotabpfn/checkpoint/dir"
    save_temp_every: int = 50
    save_perm_every: int = 5000
    max_checkpoints: int = 5
    path: Optional[str] = None
    only_load_model: bool = False

# -----------------------
# Wandb Config
# -----------------------
@dataclass
class WandbConfig:
    log: bool = True
    project: str = "TICLA"
    name: Optional[str] = "LoopedNanoTabPFN"
    id: Optional[str] = None
    dir: Optional[str] = "./wandb/"
    new_run: bool = True
    mode: str = "online"



# -----------------------
# Master Config
# -----------------------
@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)  
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
