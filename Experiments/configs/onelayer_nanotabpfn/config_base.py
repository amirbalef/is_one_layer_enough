from pathlib import Path

number_of_layers = 1
config_base = {}
config_base["model"] = "onelayer_nanotabpfn"
config_base["preprocess"] = True

config_base["model_parameters"] = {"model_path":  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/NanoTabPFN/onelayer_nanotabpfn.ckpt", "model_name":"OneLayerNanoTabPFN" }

layers_info = []
for layer in range(number_of_layers):
    layers_info.append(
            (
                layer,
                {
                "w_attn_features_on_query": 1.0,
                "w_attn_features_on_support": 1.0,
                "w_attn_items_on_query": 1.0,
                "w_attn_items_on_support": 1.0, 
                "w_mlp_on_support":1.0,
                "w_mlp_on_query": 1.0,
                },
            )
        )
config_base["model_parameters"]["layers_info"] = layers_info
