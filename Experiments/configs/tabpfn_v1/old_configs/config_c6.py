
def generate_config_list():
    def disabling_layer_config(t, m):
        config = {}
        config["model"] = "tabpfn_v1"
        config["model_name"] = f"{config['model']}_c6_{t[0]}{m}"
        config["desc"] = f"{config['model']} with disabling layer {t} {m}"

        config["preprocess"] = True

        config["model_parameters"] = {
            "N_ensemble_configurations": 1,
            "subsample_features": True,
        }
        # Dynamic layer definition
        layers_info = []
        for layer in range(12):
            if m == layer:
                pass
            else:
                layers_info.append(
                    (
                        layer,
                        {
                            "w_attn_on_query": 1.0,
                            "w_attn_on_support": 1.0,
                            "w_ffn_on_support": 1.0,
                            "w_ffn_on_query": 1.0,
                        },
                    )
                )

        config["model_parameters"]["layers_info"] = layers_info
        return config


    configs = []
    types = ["predictor"]
    for t in types:
        max_m = 12
        for m in range(max_m):
            new_config = disabling_layer_config(t, m)
            configs.append(new_config)
    return configs
