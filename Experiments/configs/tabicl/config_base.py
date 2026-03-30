from pathlib import Path


number_of_predictor_layers = 12
number_of_rows_layers = 3
number_of_col_layers = 3

config_base = {}
config_base["model"] = "tabicl"
config_base["preprocess"] = False

config_base["finetuned_decoders_path"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config_base["model"] 
config_base["separate_y_embeddings"] = False
config_base["model_parameters"] = { "n_estimators": 1, "finetuned_decoders_path": config_base["finetuned_decoders_path"]}

col_layers_info =[]
for layer in range(number_of_col_layers):
    col_layers_info.append(( layer ,  {"attn1":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0, "compute_component_contribution": False}, 
    "attn2":{"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0}, "compute_component_contribution": False, "order_needs_transpose": False } ))

row_layers_info =[]
for layer in range(number_of_rows_layers):
    row_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0, "compute_component_contribution": False, "order_needs_transpose": True} ))

predictor_layers_info =[]
for layer in range(number_of_predictor_layers):
    predictor_layers_info.append(( layer ,  {"w_attn_on_query": 1.0, "w_attn_on_support": 1.0, "w_ffn_on_support":1.0, "w_ffn_on_query": 1.0} ))


config_base["model_parameters"]["col_layers_info"] = col_layers_info
config_base["model_parameters"]["row_layers_info"] = row_layers_info
config_base["model_parameters"]["predictor_layers_info"] = predictor_layers_info
