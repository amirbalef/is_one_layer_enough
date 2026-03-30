def generate_config_list():
    config = {}
    config["model"] = "tabicl"
    config["model_name"] = f"{config['model']}_c9"
    config["desc"] = f"{config['model']}_shuffle_train_labels"
    config["shuffle_train_labels"] = True

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


    config["model_parameters"]["col_layers_info"] = col_layers_info
    config["model_parameters"]["row_layers_info"] = row_layers_info
    config["model_parameters"]["predictor_layers_info"] = predictor_layers_info
    return [config]
    