from pathlib import Path

number_of_layers = 12
config_base = {}
config_base["model"] = "tabpfn_v1"
config_base["preprocess"] = True

config_base["finetuned_decoders_path"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config_base["model"] 
config_base["separate_y_embeddings"] = False

config_base["model_parameters"] = {
        "N_ensemble_configurations": 1,
        "subsample_features": True,
        "finetuned_decoders_path": config_base["finetuned_decoders_path"],
        }

layers_info = []
for layer in range(number_of_layers):
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
config_base["model_parameters"]["layers_info"] = layers_info
