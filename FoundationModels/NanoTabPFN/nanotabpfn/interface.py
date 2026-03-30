import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device
import copy

def init_model_from_state_dict_file(file_path, model_name):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    if file_path.endswith('.ckpt'):
        # convert from lightning checkpoint to state dict
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        state_dict = {
            'architecture': checkpoint['config'],
            'model': checkpoint['state_dict']
        }
    else:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    if model_name == "NanoTabPFNModel":
        model = NanoTabPFNModel(
            num_attention_heads=state_dict['architecture']['num_attention_heads'],
            embedding_size=state_dict['architecture']['embedding_size'],
            mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
            num_layers=state_dict['architecture']['num_layers'],
            num_outputs=state_dict['architecture']['num_outputs'],
        )
    elif  model_name == "OneLayerNanoTabPFN":
        from .custom_models.onelayer_nanotabpfn import OneLayerNanoTabPFN
        model = OneLayerNanoTabPFN(
            num_attention_heads=state_dict['architecture']['num_attention_heads'],
            embedding_size=state_dict['architecture']['embedding_size'],
            mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
            num_layers=state_dict['architecture']['num_layers'],
            num_outputs=state_dict['architecture']['num_outputs'],
        )

    
    elif model_name == "LoopedNanoTabPFN":
        from .custom_models.looped_nanotabpfn import LoopedNanoTabPFN
        model = LoopedNanoTabPFN(
            num_attention_heads=state_dict['architecture']['num_attention_heads'],
            embedding_size=state_dict['architecture']['embedding_size'],
            mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
            num_layers=state_dict['architecture']['num_layers'],
            num_outputs=state_dict['architecture']['num_outputs'],
        )
        
    else:
        raise NotImplementedError
    model.load_state_dict(state_dict['model'])
    return model

# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x

def to_numeric(x):
    return x.apply(pd.to_numeric, errors='coerce').to_numpy()

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(to_numeric)), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model_path: NanoTabPFNModel|str|None = None, model_name: str|None = "NanoTabPFNModel" , device: None|str|torch.device = None, num_mem_chunks: int =8 , layers_info = None, finetuned_decoders_path : str | None = None, random_state: int|None = None):
        if device is None:
            device = get_default_device()
        if model_path is None:
            model_path = 'checkpoints/nanotabpfn.pth'
            if not os.path.isfile(model_path):
                os.makedirs("checkpoints", exist_ok=True)
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth')
                with open(model_path, 'wb') as f:
                    f.write(response.content)
        if isinstance(model_path, str):
            model = init_model_from_state_dict_file(model_path, model_name)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        self.layers_info = layers_info
        self.random_state = random_state
        self.finetuned_decoders_path = finetuned_decoders_path
        if self.finetuned_decoders_path is not None:
            self.load_finetuned_decoders()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = len(np.unique(y_train))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_with_proba(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1), predicted_probabilities


    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks, layers_info = self.layers_info).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').numpy()

    def get_all_contribution_scores(self):
        contribution_scores = {}
        for l, info in self.layers_info:
            layer = self.model.transformer_encoder.transformer_blocks[l]
            contribution_scores[f'layer_{l}'] = layer.component_contribution_scores
        return contribution_scores
    
    def get_all_layers_embeddings(self): #todo fix for n_estimators>1
        embeddings = []
        for l, info in self.layers_info:
            layer = self.model.transformer_encoder.transformer_blocks[l]
            embeddings.append(layer.out_embeddings.squeeze(0))
        return embeddings

    def get_decoder(self):
        return self.model.decoder

    @torch.no_grad()
    def get_all_layers_predictions(self, embeddings=None, decoder_type = "default"):
        if embeddings is None:
            embeddings = self.get_all_layers_embeddings()
        all_layer_prediction = []
        all_layer_predict_proba = []
        for l, emb in enumerate(embeddings):
            device = next(self.model.parameters()).device
            if decoder_type ==  "default":
                decoder = self.get_decoder()
            if decoder_type ==  "finetuned":
                decoder = self.finetuned_decoders[l].to(device)

            out = decoder(emb[:, -1, :].to(device).float())
            out = out[:, :self.num_classes]
            p = F.softmax(out, dim=1).to('cpu').numpy()
            all_layer_predict_proba.append(p)
            y = np.argmax(p, axis=-1)
            all_layer_prediction.append(y)
        return all_layer_prediction, all_layer_predict_proba


    def get_feature_encoder_embeddings(self): #todo fix for n_estimators>1
        return [self.model.transformer_encoder.transformer_blocks[self.layers_info[0][0]].in_embeddings.squeeze(0)]

    def get_feature_encoder_predictions(self):
        return self.get_all_layers_predictions(embeddings=self.get_feature_encoder_embeddings())

    def load_finetuned_decoders(self):
        self.finetuned_decoders =[]
        for l, info in self.layers_info:
            decoder = copy.deepcopy(self.get_decoder())
            model_state_dict = torch.load(self.finetuned_decoders_path  + f"/decoder_layer_{l}.pth", map_location=torch.device('cpu'))
            decoder.load_state_dict(model_state_dict)
            self.finetuned_decoders.append(decoder)
        return  self.finetuned_decoders

    def model_inference(self,  X, y , support_size):
        output = self.model._forward((X.to(torch.float).to(self.device) , y.to(torch.float).to(self.device) ), support_size, num_mem_chunks=self.num_mem_chunks, layers_info = self.layers_info)
        return output

class NanoTabPFNRegressor():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, dist: FullSupportBarDistribution|str|None = None, device: str|torch.device|None = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = 'checkpoints/nanotabpfn_regressor.pth'
            dist = 'checkpoints/nanotabpfn_regressor_buckets.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
