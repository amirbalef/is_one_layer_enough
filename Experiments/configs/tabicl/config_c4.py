from .config_base import config_base
import copy

def generate_config_list():
    conf = config_base.copy()

    def custom_layer_config(t, m):
        config =  copy.deepcopy(config_base)
        config["model_name"] = f"{config['model']}_c4_{t[0]}{m}"
        config["desc"] = f"{config['model']} with swaping layer {m} and {m+1} for {t} layers"
        if t == "col":
            col_layers_info = config["model_parameters"]["col_layers_info"]
            col_layers_info[m], col_layers_info[m+1] = col_layers_info[m+1], col_layers_info[m]
            config["model_parameters"]["col_layers_info"] = col_layers_info
        elif t == "row":
            row_layers_info = config["model_parameters"]["row_layers_info"]
            row_layers_info[m], row_layers_info[m+1] = row_layers_info[m+1], row_layers_info[m]
            config["model_parameters"]["row_layers_info"] = row_layers_info
        elif t == "predictor":
            predictor_layers_info = config["model_parameters"]["predictor_layers_info"]
            predictor_layers_info[m], predictor_layers_info[m+1] = predictor_layers_info[m+1], predictor_layers_info[m]
            config["model_parameters"]["predictor_layers_info"] = predictor_layers_info
        return config

    configs = []
    types = ["col", "row", "predictor"]
    for t in types:
        if t == "col":
            max_m = conf["model_parameters"]["col_layers_info"] .__len__() - 1
        elif t == "row":
            max_m = conf["model_parameters"]["row_layers_info"] .__len__() - 1
        else:
            max_m = conf["model_parameters"]["predictor_layers_info"].__len__() - 1
        for m in range(max_m):
            new_config = custom_layer_config(t, m)
            configs.append(new_config)
    return configs