from .config_base import config_base
import copy

def generate_config_list():
    conf = config_base.copy()

    def custom_layer_config(t, m):
        config =  copy.deepcopy(config_base)
        config["model_name"] = f"{config['model']}_c3_{t[0]}{m}"
        config["desc"] = f"{config['model']} with repeating layer {t} {m}"
        if t == "col":
            col_layers_info = config["model_parameters"]["col_layers_info"]
            col_layers_info.insert(m, col_layers_info[m]) 
            config["model_parameters"]["col_layers_info"] = col_layers_info
        elif t == "row":
            row_layers_info = config["model_parameters"]["row_layers_info"]
            row_layers_info.insert(m, row_layers_info[m])
            config["model_parameters"]["row_layers_info"] = row_layers_info
        elif t == "predictor":
            predictor_layers_info = config["model_parameters"]["predictor_layers_info"]
            predictor_layers_info.insert(m, predictor_layers_info[m])
            config["model_parameters"]["predictor_layers_info"] = predictor_layers_info
        return config

    configs = []
    types = ["col", "row", "predictor"]
    for t in types:
        if t == "col":
            max_m = conf["model_parameters"]["col_layers_info"] .__len__()
        elif t == "row":
            max_m = conf["model_parameters"]["row_layers_info"] .__len__() 
        else:
            max_m = conf["model_parameters"]["predictor_layers_info"].__len__()
        for m in range(max_m):
            new_config = custom_layer_config(t, m)
            configs.append(new_config)
    return configs
