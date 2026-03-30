def generate_config_list():

    def swap_layer_config(t, m):
        config = {}
        config["model"] = "tabicl"
        config["model_name"] = f"{config['model']}_c1_{t[0]}{m}"
        config["desc"] = f"{config['model']} with swaping layer {m} and {m+1} for {t} layers"

        config["preprocess"] = False

        config["model_parameters"] = { "n_estimators": 1}
        # Dynamic layer definition

        col_layers_info =[]
        for layer in range(3):
            col_layers_info.append(( layer ,  {"attn1":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}, 
            "attn2":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}} ))

        row_layers_info =[]
        for layer in range(3):
            row_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))

        predictor_layers_info =[]
        for layer in range(12):
            predictor_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))

        if t == "col":
            if m < 2:
                col_layers_info[m], col_layers_info[m+1] = col_layers_info[m+1], col_layers_info[m]
            else:
                print(f"Invalid layer index {m} for col layers. Skipping swap.")
        elif t == "row":
            if m < 2:
                row_layers_info[m], row_layers_info[m+1] = row_layers_info[m+1], row_layers_info[m]
            else:
                print(f"Invalid layer index {m} for row layers. Skipping swap.")
        elif t == "predictor":
            if m < 11:
                predictor_layers_info[m], predictor_layers_info[m+1] = predictor_layers_info[m+1], predictor_layers_info[m]
            else:
                print(f"Invalid layer index {m} for predictor layers. Skipping swap.")

        config["model_parameters"]["col_layers_info"] = col_layers_info
        config["model_parameters"]["row_layers_info"] = row_layers_info
        config["model_parameters"]["predictor_layers_info"] = predictor_layers_info
        return config

    configs = []
    types = ["col", "row", "predictor"]
    for t in types:
        if t == "col":
            max_m = 2
        elif t == "row":
            max_m = 2
        else:
            max_m = 11
        for m in range(max_m):
            new_config = swap_layer_config(t, m)
            configs.append(new_config)
    return configs
    