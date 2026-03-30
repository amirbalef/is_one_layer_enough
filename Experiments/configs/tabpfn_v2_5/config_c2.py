from .config_base import config_base
import copy

def generate_config_list():
    conf = config_base.copy()

    def custom_layer_config(t, m):
        config =  copy.deepcopy(config_base)
        config["model_name"] = f"{config['model']}_c2_{t[0]}{m}"
        config["desc"] = f"{config['model']} with skipping layer {t} {m}"
        layers_info = config["model_parameters"]["layers_info"]
        del layers_info[m]
        config["model_parameters"]["layers_info"] = layers_info
        return config
    configs = []
    types = ["predictor"]
    for t in types:
        max_m = conf["model_parameters"]["layers_info"].__len__()
        for m in range(max_m):
            new_config = custom_layer_config(t, m)
            configs.append(new_config)
    return configs