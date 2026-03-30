## Overview
How each of the transformer layers behaves at inference time, via `layers_info`

---

## Directory Structure

The config assumes the following directory layout relative to this file (4 levels up from the config's location):

```
<project_root>/
└── FoundationModels/
    └── weights/
        ├── Limix/
        │   └── LimiX-2M.ckpt          # Main model checkpoint
        └── extras/
            └── limix_2m/               # Fine-tuned decoder weights
```
---

## Setup

### Download Weights

Place the model checkpoint and fine-tuned decoder weights in the paths described above, for example:

- **Main checkpoint:** `FoundationModels/weights/Limix/LimiX-2M.ckpt`
- **Fine-tuned decoders:** `FoundationModels/weights/extras/limix_2m/`



## Modifying `layers_info`

`layers_info` is a list of `(layer_index, weights_dict)` tuples — one per transformer layer. It gives you fine-grained control over how much each layer contributes during the forward pass.

### Structure

```python
layers_info = [
    (
        layer_index,          # int: 0 to number_of_layers - 1
        {
            "w_attn_sequence_on_query":    1.0,  # Sequence attention weight on query set
            "w_attn_sequence_on_support":  1.0,  # Sequence attention weight on support set
            "w_mlp_on_support":            1.0,  # MLP weight on support set
            "w_mlp_on_query":              1.0,  # MLP weight on query set
            "w_attn_features_on_query":    1.0,  # Feature attention weight on query set
            "w_attn_features_on_support":  1.0,  # Feature attention weight on support set
        }
    ),
    ...
]
```

### Weight Parameters

| Parameter | Description |
|-----------|-------------|
| `w_attn_sequence_on_query` | Scales sequence-level self-attention applied to the **query** inputs |
| `w_attn_sequence_on_support` | Scales sequence-level self-attention applied to the **support** inputs |
| `w_mlp_on_support` | Scales the MLP block output on **support** inputs |
| `w_mlp_on_query` | Scales the MLP block output on **query** inputs |
| `w_attn_features_on_query` | Scales feature-level attention on **query** inputs |
| `w_attn_features_on_support` | Scales feature-level attention on **support** inputs |

All weights default to `1.0` (full contribution). Setting a weight to `0.0` disables that component for that layer.

### Example: Disable MLP in early layers

```python
for layer in range(number_of_layers):
    w_mlp = 0.0 if layer < 4 else 1.0
    layers_info.append((
        layer,
        {
            "w_attn_sequence_on_query":   1.0,
            "w_attn_sequence_on_support": 1.0,
            "w_mlp_on_support":           w_mlp,
            "w_mlp_on_query":             w_mlp,
            "w_attn_features_on_query":   1.0,
            "w_attn_features_on_support": 1.0,
        }
    ))
```

### Example: Scale down attention in later layers

```python
for layer in range(number_of_layers):
    w_attn = 1.0 if layer < 8 else 0.5
    layers_info.append((
        layer,
        {
            "w_attn_sequence_on_query":   w_attn,
            "w_attn_sequence_on_support": w_attn,
            "w_mlp_on_support":           1.0,
            "w_mlp_on_query":             1.0,
            "w_attn_features_on_query":   w_attn,
            "w_attn_features_on_support": w_attn,
        }
    ))
```