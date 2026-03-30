
def generate_config_list():
    config = {}
    config["model"] = "tabpfn_v1"
    config["model_name"] = f"{config['model']}_c11"
    config["desc"] = f"{config['model']} layerwise_probing"
    config["layerwise_probing"] = True
    config["embedding_distances"] = True
    config["full_eval"] = True

    config["preprocess"] = True

    config["model_parameters"] = {
        "N_ensemble_configurations": 1,
        "subsample_features": True,
    }
    # Dynamic layer definition
    layers_info = []
    for layer in range(12):
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
    return [config]