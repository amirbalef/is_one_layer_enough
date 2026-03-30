from .config_base import config_base
from pathlib import Path

def generate_config_list():
    config = config_base.copy()
    config["model_name"] = f"{config['model']}"
    config["desc"] = f"{config['model']}_finetuned_decoder"
    
    config["separate_y_embeddings"] = False
    config["batch_first"] = False
    config["decoder_pre_norm"] = True


    config["prior_batch_size"] = 512
    config["prior_data_dir"] =  str(Path(__file__).resolve().parents[3]) + "/Pretraining/TICLA/workdir/nanotabpfn/prior/dataset"
    config["training_max_steps"] = 100
    config["micro_batch_size"] = 64
    config["training_batch_size"] = 256

    config["saving_dir"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config["model"] + "_new" 
    config["save_every_step"] = 100

    config["wandb_mode"] = "online"

    for item in config["model_parameters"]["predictor_layers_info"]:
        item[1]["compute_component_contribution"] = False
    
    for item in config["model_parameters"]["row_layers_info"]:
        item[1]["compute_component_contribution"] = False

    for item in config["model_parameters"]["col_layers_info"]:
        item[1]["attn1"]["compute_component_contribution"] = False
        item[1]["attn2"]["compute_component_contribution"] = False
        item[1]["compute_component_contribution"] = False
    return [config]