
from .config_base import config_base

def generate_config_list():
    config = config_base.copy()
    config["model_name"] = f"{config['model']}_c1"
    config["desc"] = f"{config['model']} layerwise_probing_decoder (y embeddings)"
    config["layerwise_probing"] = True
    config["embedding_distances"] = False
    config["half_eval"] = True
    for item in config["model_parameters"]["predictor_layers_info"]:
        item[1]["compute_component_contribution"] = True
    for item in config["model_parameters"]["row_layers_info"]:
        item[1]["compute_component_contribution"] = True
    for item in config["model_parameters"]["col_layers_info"]:
        item[1]["attn1"]["compute_component_contribution"] = False
        item[1]["attn2"]["compute_component_contribution"] = True
        item[1]["compute_component_contribution"] = True
        
    return [config]