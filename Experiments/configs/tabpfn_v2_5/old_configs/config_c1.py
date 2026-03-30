
from pathlib import Path
import inspect
import tabpfn_v2_5
from tabpfn_v2_5.model_loading import ModelSource


def generate_config_list():
    def swap_layer_config(t, m):
        config = {}
        config["model"] = "tabpfn_v2_5"
        config["model_name"] = f"{config['model']}_c1_{t[0]}{m}"
        config["desc"] = f"{config['model']} with swaping layer {m} and {m+1} for {t} layers"

        config["preprocess"] = False

        config["model_parameters"] = { "n_estimators": 1, "ignore_pretraining_limits": True,
        "model_path" : os.path.dirname(inspect.getfile(tabpfn_v2_5))  + "/weights/" + ModelSource.get_classifier_v2_5().default_filename }

        # Dynamic layer definition
        layers_info = []
        for layer in range(24):
            layers_info.append(
                (
                    layer,
                    {
                    "w_attn_between_features_on_query": 1.0,
                    "w_attn_between_features_on_support": 1.0,
                    "w_attn_between_items_on_query": 1.0,
                    "w_attn_between_items_on_support": 1.0, 
                    "w_mlp_on_support":1.0,
                    "w_mlp_on_query": 1.0,
                    "w_second_mlp_on_support":1.0,
                    "w_second_mlp_on_query": 1.0
                    },
                )
            )
        if t == "predictor":
            if m < 23:
                layers_info[m], layers_info[m + 1] = layers_info[m + 1], layers_info[m]
            else:
                print(f"Invalid layer index {m} for predictor layers. Skipping swap.")

        config["model_parameters"]["layers_info"] = layers_info
        return config
        
    configs = []
    types = ["predictor"]
    for t in types:
        max_m = 23
        for m in range(max_m):
            new_config = swap_layer_config(t, m)
            configs.append(new_config)
    return configs
