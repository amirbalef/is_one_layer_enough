from .config_base import config_base
import copy

def generate_config_list():
    conf = config_base.copy()

    def custom_layer_config(t, m):
        config =  copy.deepcopy(config_base)
        config["model_name"] = f"{config['model']}_c5_{t[0]}{m}"
        config["desc"] = f"{config['model']} with swaping layer {m-1} and {m+1} for {t} layers"
        layers_info = config["model_parameters"]["layers_info"]
        layers_info[m-1], layers_info[m+1] = layers_info[m+1], layers_info[m-1]
        config["model_parameters"]["layers_info"] = layers_info
        return config
    configs = []
    types = ["predictor"]
    for t in types:
        max_m = conf["model_parameters"]["layers_info"].__len__() - 1
        for m in range(1, max_m):
            new_config = custom_layer_config(t, m)
            configs.append(new_config)
    return configs