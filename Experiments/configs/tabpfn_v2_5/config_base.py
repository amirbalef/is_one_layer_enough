from pathlib import Path
from tabpfn_v2_5.model_loading import ModelSource

number_of_layers = 24
config_base = {}
config_base["model"] = "tabpfn_v2_5"
config_base["preprocess"] = False
config_base["separate_y_embeddings"] = True


config_base["finetuned_decoders_path"] =  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/extras/" + config_base["model"] 

config_base["model_parameters"] = { "n_estimators": 1, "ignore_pretraining_limits": True,
    "model_path" :  str(Path(__file__).resolve().parents[3]) + "/FoundationModels/weights/TabPFN_v2_5/"+ ModelSource.get_classifier_v2_5().default_filename,
    "finetuned_decoders_path": config_base["finetuned_decoders_path"],
     }

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
                "w_second_mlp_on_support":1.0,
                "w_second_mlp_on_query": 1.0
                },
            )
        )
config_base["model_parameters"]["layers_info"] = layers_info
