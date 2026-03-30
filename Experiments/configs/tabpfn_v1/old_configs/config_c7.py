
def generate_config_list():

    def reapting_layer_config(t, m, n =2):
        config = {}
        config["model"] = "tabpfn_v1"
        config["model_name"] = f"{config['model']}_c7_{t[0]}{m}"
        config["desc"] = f"{config['model']} with n more times repeating layer {t} {m}"

        config["preprocess"] = True

        config["model_parameters"] = {
            "N_ensemble_configurations": 1,
            "subsample_features": True,
        }
        # Dynamic layer definition
        layers_info = []
        for layer in range(12):
            if m == layer:
                for _ in range(n+1):
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
            new_config = reapting_layer_config(t, m)
            configs.append(new_config)
    return configs
