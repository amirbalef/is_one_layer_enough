# TabFM-insights

A research codebase for understanding inference dynamics in tabular
in-context learning models (TabPFN, TabICL, LimiX, …) across OpenML benchmarks.

---

## Multiclass Results

Here we provide the results for multiclass problems. We use all $8$ multiclass tasks available in TabArena (excluding two datasets due to memory constraints).

### Exp 1. Embedding Similarity


<p align="center">
  <img src="Assets/Multiclass/c1_TabArena_similarity-1.png" width="75%" />
  <img src="Assets/Multiclass/c1_TabArena_similarity_colorbar-1.png" width="8.5%" />
</p>


---

### Exp 2. Separation Gap

<p align="center">
  <img src="Assets/Multiclass/c0_separation_gap_TabArena_cosine-1.png" width="85%" />
  <img src="Assets/Multiclass/c0_separation_gap_TabArena_cosine_legend-1.png" width="65%" />
</p>

---

### Exp 3. Probing Classifier

<p align="center">
  <img src="Assets/Multiclass/c1_TabArena_logistic_regression_balanced_accuracy-1.png" width="90%" />
  <img src="Assets/Multiclass/c1_TabArena_logistic_regression_balanced_accuracy_colorbar-1.png" width="8%" /> &nbsp;&nbsp;
</p>

---

### Exp 4. Tabular Logit Lens

![Early Exit AUC/ACC](Assets/Multiclass/c0_early_exit_TabArena_auc_acc-1.png)
![Early Exit AUC/ACC Legend](Assets/Multiclass/c0_early_exit_TabArena_auc_acc_legend-1.png)


---

### Exp 5. Layer Interventions

![Layer Interventions TabArena](Assets/Multiclass/layer_interventions_TabArena-1.png)
![Layer Interventions TabArena Legend](Assets/Multiclass/layer_interventions_TabArena_legend-1.png)

---

### Exp 6. Self Repair (Skipping Layer)

![Skipping Layer All](Assets/Multiclass/c2_skipping_layer_all-1.png)

---

### POC: Is One Layer Enough?

<p align="center">
  <img src="Assets/Multiclass/looped_transformer_early_exit_both_c0-1.png" width="55%" />
  <img src="Assets/Multiclass/looped_transformer_per_dataset_c0_roc_auc-1.png" width="42%" />
</p>

In addition to multiclass tasks available in TabArena, we evaluated the models on $20$ multiclass tasks available in OpenML-CC18.


## Repository structure

```
.
├── Experiments/                    # Main experiment code
│   ├── main.py                     # the main code for saving the results.
│   ├── fine_tuning_exp.py          # the code for tabular logit lens (pretrained individual decoders)
│   ├── util.py                     # some utilities (metrics, preprocessing, probing, …)
│   ├── configs/                    # Per-model experiment configs.
│   │   ├── tabpfn_v1/
│   │   ├── tabpfn_v2/
│   │   ├── tabpfn_v2_5/
│   │   ├── tabicl/
│   │   ├── limix_2m/
│   │   ├── limix_16m/
│   │   ├── nanotabpfn/
│   │   └── ...
│   ├── plots/                      # plots the results
│   └── results/                    # Output results (gitignored)
│
├── FoundationModels/               # Edited model packages
│   ├── TabPFN_v1/
│   ├── TabPFN_v2/
│   ├── TabPFN_v2_5/
│   ├── TabPFN_mix/
│   ├── TabICL/
│   ├── Limix/
│   ├── Mitra/
│   ├── NanoTabPFN/
│   └── weights/                    # Pre-trained model weights and individual decoder weights
│
├── Notebooks/                      # prototyping notebooks
│
├── Pretraining/                    # Prior data generation and pretraining NanoTabPFN
│
├── requirements.txt
└── README.md
```


# Dependencies

## 1. Install UV

Install UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
```

```Bash
# Create a venv (stored locally in the project folder)
uv venv --python 3.11
source .venv/bin/activate

# Install and run
uv pip install -r requirements.txt
# example uv run --frozen my_script.py # this is faster, but assumes your .venv is in sync with uv.lock
```

### 2. Install model packages

Each model lives under `FoundationModels/` (and better to be installed in editable mode):

```bash
pip install -e FoundationModels/TabPFN_v1
pip install -e FoundationModels/TabPFN_v2
pip install -e FoundationModels/TabPFN_v2_5
pip install -e FoundationModels/TabICL
pip install -e FoundationModels/Limix
pip install -e FoundationModels/NanoTabPFN
```

Place the model checkpoint and fine-tuned decoders weights in `FoundationModels/weights`, to use the models please check the README in  `FoundationModels/`


### 3. Install the pretraining and prior generation package

```bash
pip install -e Pretraining/TICLA
```


## Example: Running the experiment

change directory to `Experiments/` .

```bash
cd Experiments
```

### Single task

```bash
python main.py \
  --task_id 363619 \
  --config tabpfn_v2.config_c0 \
  --output_root_dir ./results
```


## Example: Pretraining individual decoders

Trains one individual decoder head per transformer layer on synthetic prior
data.

```bash
cd Experiments
python fine_tuning_exp.py \
  --config tabpfn_v1.config_finetuning_decoder
```

---

# Experiment Configs

| Config | Description |
|--------|-------------|
| `c0`   | Default model (baseline) |
| `c1`   | Representation analysis — probes hidden states at each layer |
| `c2`   | Layer skipping — omits a layer during inference |
| `c3`   | Layer repetition — repeats a layer during inference |
| `c4`   | Layer swapping — swap neighboring layers order during inference |
| `finetuning_decoder` | Trains individual decoder heads per layer on synthetic prior data |