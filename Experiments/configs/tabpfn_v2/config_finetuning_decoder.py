from .config_base import config_base
from pathlib import Path

def generate_config_list():
    config = config_base.copy()
    config["model_name"] = f"{config['model']}"
    config["desc"] = f"{config['model']}_finetuned_decoder"
    
    config["separate_y_embeddings"] = True
    config["batch_first"] = False

    config["prior_batch_size"] = 512
    config["prior_data_dir"] =  str(Path(__file__).resolve().parents[3]) + "/Pretraining/TICLA/workdir/nanotabpfn/prior/dataset"
    config["training_max_steps"] = 200
    config["micro_batch_size"] = 64
    config["training_batch_size"] = 8

    config["saving_dir"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config["model"] + "_fast"
    config["save_every_step"] = 10

    config["wandb_mode"] = "online"

    del config["finetuned_decoders_path"]
    del config["model_parameters"]["finetuned_decoders_path"]
    for item in config["model_parameters"]["layers_info"]:
        item[1]["compute_component_contribution"] = False
    
    return [config]