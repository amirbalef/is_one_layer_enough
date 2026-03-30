def generate_config_list():
    config = {}
    config["model"] = "tabpfn_v2"
    config["model_name"] = f"{config['model']}_c9"
    config["desc"] = f"{config['model']}_shuffle_train_labels"
    config["shuffle_train_labels"] = True

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

    config["model_parameters"]["layers_info"] = layers_info
    return [config]