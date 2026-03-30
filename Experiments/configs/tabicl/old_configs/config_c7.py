def generate_config_list():

    def reapting_layer_config(t, m, n = 2):
        config = {}
        config["model"] = "tabicl"
        config["model_name"] = f"{config['model']}_c7_{t[0]}{m}"
        config["desc"] = f"{config['model']} with n times reapting layer {t} {m}"

        config["preprocess"] = False

        config["model_parameters"] = { "n_estimators": 1}
        # Dynamic layer definition

        col_layers_info =[]
        for layer in range(3):
            if m == layer and t == "col":
                for _ in range(n+1):
                    col_layers_info.append(( layer ,  {"attn1":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}, 
                "attn2":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}} ))
            else:
                col_layers_info.append(( layer ,  {"attn1":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}, 
                "attn2":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}} ))

        row_layers_info =[]
        for layer in range(3):
            if m == layer and t == "row":
                for _ in range(n+1):
                    row_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))
            else:
                row_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))

        predictor_layers_info =[]
        for layer in range(12):
            if m == layer and t == "predictor":
                for _ in range(n+1):
                    predictor_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))
            else:
                predictor_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))

        config["model_parameters"]["col_layers_info"] = col_layers_info
        config["model_parameters"]["row_layers_info"] = row_layers_info
        config["model_parameters"]["predictor_layers_info"] = predictor_layers_info
        return config

    configs = []
    types = ["col", "row", "predictor"]
    for t in types:
        if t == "col":
            max_m = 3
        elif t == "row":
            max_m = 3
        else:
            max_m = 12
        for m in range(max_m):
            new_config = reapting_layer_config(t, m)
            configs.append(new_config)
    return configs
    