def generate_config_list():
    config = {}
    config["model"] = "tabpfn_v2_5"
    config["model_name"] = f"{config['model']}_c10"
    config["desc"] = f"{config['model']} layerwise_probing_decoder"
    config["layerwise_probing"] = True
    config["embedding_distances"] = True

    config["preprocess"] = False

    config["model_parameters"] = { "n_estimators": 1, "ignore_pretraining_limits": True}
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

    config["model_parameters"]["layers_info"] = layers_info
    return [config]