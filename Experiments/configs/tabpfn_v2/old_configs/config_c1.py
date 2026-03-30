
def generate_config_list():
    def swap_layer_config(t, m):
        config = {}
        config["model"] = "tabpfn_v2"
        config["model_name"] = f"{config['model']}_c1_{t[0]}{m}"
        config["desc"] = f"{config['model']} with swaping layer {m} and {m+1} for {t} layers"

        config["preprocess"] = False

        config["model_parameters"] = { "n_estimators": 1, "ignore_pretraining_limits": True}
        # Dynamic layer definition
        layers_info = []
        for layer in range(12):
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
            if m < 11:
                layers_info[m], layers_info[m + 1] = layers_info[m + 1], layers_info[m]
            else:
                print(f"Invalid layer index {m} for predictor layers. Skipping swap.")

        config["model_parameters"]["layers_info"] = layers_info
        return config
        
    configs = []
    types = ["predictor"]
    for t in types:
        max_m = 11
        for m in range(max_m):
            new_config = swap_layer_config(t, m)
            configs.append(new_config)
    return configs
