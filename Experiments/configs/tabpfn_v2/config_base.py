from pathlib import Path

number_of_layers = 12
config_base = {}
config_base["model"] = "tabpfn_v2"
config_base["preprocess"] = False
config_base["separate_y_embeddings"] = True

config_base["finetuned_decoders_path"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config_base["model"] 

config_base["model_parameters"] = { "n_estimators": 1, "ignore_pretraining_limits": True, "finetuned_decoders_path": config_base["finetuned_decoders_path"]}

layers_info = []
for layer in range(number_of_layers):
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
                },
            )
        )
config_base["model_parameters"]["layers_info"] = layers_info
