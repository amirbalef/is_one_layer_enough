
from importlib import import_module
from typing import Dict, Optional
import torch

def load_model_config(config_name: str) -> Dict:
    """Retrieve model configuration by name."""
    module = import_module( f"ticla.model_configs.{config_name}")
    Config = getattr(module, "Config")
    return Config()


def load_model_class(model_config):
    module = import_module(
    f"ticla.models.{model_config.general.name}."
            f"{model_config.general.variant}"
        )
    Model = getattr(module, model_config.general.class_name)
    return Model


def load_model_inference(model_config, inference_config: Optional[Dict] = None, device: str = "cpu", use_amp: bool = False, verbose: bool = False):
    inference_config_ = None
    if model_config.general.name == "tabicl":
        from ticla.models.tabicl.inference_config import InferenceConfig
        init_config = {
            "COL_CONFIG": {"device": device, "use_amp": use_amp, "verbose": verbose},
            "ROW_CONFIG": {"device": device, "use_amp": use_amp, "verbose": verbose},
            "ICL_CONFIG": {"device": device, "use_amp": use_amp, "verbose": verbose},
        }
        # If None, default settings in InferenceConfig
        if inference_config is None:
            inference_config_ = InferenceConfig()
            inference_config_.update_from_dict(init_config)
        # If dict, update default settings
        elif isinstance(inference_config, dict):
            inference_config_ = InferenceConfig()
            for key, value in inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            inference_config_.update_from_dict(init_config)
        # If InferenceConfig, use as is
        else:
            inference_config_ = inference_config
    return inference_config_
            
def load_model_weights(model, model_config, device: str = "cpu"):
    """Load model weights from specified path."""
    weights_path = model_config.general.weights_path
    if weights_path:
        if weights_path.split(".")[-1] == "ckpt":
            state_dict = torch.load(weights_path, map_location=torch.device(device))["state_dict"]
            model.load_state_dict(state_dict)
        elif weights_path.split(".")[-1] == "pth":
            state_dict = torch.load(weights_path, map_location=torch.device(device))['model']
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported weights file format: {weights_path}")
    return model


def build_model(config: Optional[Dict] = None) -> None:
        """Build and initialize the TabICL model."""
        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
        }
        model = TabICL(**self.model_config)
        model.to(device=self.config.device)
        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")
            
        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")
        
        return model

        

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model