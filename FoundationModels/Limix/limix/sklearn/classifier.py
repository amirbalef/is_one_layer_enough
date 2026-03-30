from __future__ import annotations

import warnings
from pathlib import Path
from packaging import version
from typing import Optional, List, Dict
import numpy as np
import torch
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from .preprocessing import TransformToNumerical
from limix.inference.predictor import LimiXPredictor
import os
import copy

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class LimiXClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str | torch.device] = None,
        random_state: int | None = 0,
        inference_config :Optional[str | Path] = None,
        layers_info: Optional[Dict[str, List[tuple]]] = None,
        task_type: str =  "Classification",
        finetuned_decoders_path : str | None = None,
    ):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)

        self.random_state = random_state
        self.layers_info = layers_info
        self.inference_config = inference_config if inference_config is not None else str(Path(__file__).resolve().parent.parent / 'config'/'cls_default_noretrieval_1.json')
        self.model_path = model_path if model_path is not None else  str(Path(__file__).resolve().parent.parent.parent.parent) + "/weights/Limix/LimiX-16M.ckpt"
        self.task_type = task_type
        self.extra_preprocessing = False
        self.predictor = None
        self.finetuned_decoders_path = finetuned_decoders_path
        if self.finetuned_decoders_path is not None:
            self.load_finetuned_decoders()


    def _more_tags(self):
        """Mark classifier as non-deterministic to bypass certain sklearn tests."""
        return dict(non_deterministic=True)
    

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    @torch.no_grad()
    def fit(self, X, y):
        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
        else:
            X, y = self._validate_data(X, y, dtype=None, skip_check_array=True)

        check_classification_targets(y)
        self.predictor = LimiXPredictor(device=self.device, model_path=self.model_path, inference_config=self.inference_config, seed =self.random_state, layers_info=self.layers_info)

        if self.extra_preprocessing:
            # Encode class labels
            self.y_encoder_ = LabelEncoder()
            y = self.y_encoder_.fit_transform(y)
            self.classes_ = self.y_encoder_.classes_
            self.n_classes_ = len(self.y_encoder_.classes_)

            if self.n_classes_ > self.model_.max_classes and not self.use_hierarchical:
                raise ValueError(
                    f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                    f"natively supported by the model. Consider enabling hierarchical classification."
                )

            if self.n_classes_ > self.model_.max_classes and self.verbose:
                print(
                    f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                    f"natively supported by the model. Therefore, hierarchical classification is used."
                )

            #  Transform input features
            self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
            X = self.X_encoder_.fit_transform(X)

        self.X = X
        self.y = y

        return self

    @torch.no_grad()
    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError(f"The provided input X is one-dimensional. Reshape your data.")
        # Preserve DataFrame structure to retain column names and types for correct feature transformation
        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)

        if self.extra_preprocessing:

            X = self.X_encoder_.transform(X)

        # Normalize probabilities to sum to 1
        return self.predictor.predict(self.X, self.y, X, task_type=self.task_type)

    @torch.no_grad()
    def predict(self, X):
        """Predict class labels for test samples.

        Uses predict_proba to get class probabilities and returns the class with
        the highest probability for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels for each test sample.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        if self.extra_preprocessing:
            return self.y_encoder_.inverse_transform(y)
        y = self.predictor.classes.take(np.asarray(y, dtype=np.intp))
        return y

    @torch.no_grad()
    def predict_with_proba(self, X):
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        if self.extra_preprocessing:
            return self.y_encoder_.inverse_transform(y)
        y = self.predictor.classes.take(np.asarray(y, dtype=np.intp))
        return y, proba


    @staticmethod
    def softmax(x, axis: int = -1, temperature: float = 0.9):
        x = x / temperature
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        # Compute softmax
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def get_all_contribution_scores(self):
        contribution_scores = {}
        for l, info in self.layers_info:
            layer = self.predictor.model.transformer_encoder.layers[l]
            contribution_scores[f'layer_{l}'] = layer.component_contribution_scores
        return contribution_scores

    def get_feature_encoder_embeddings(self): #todo fix for n_estimators>1
        return [self.predictor.model.transformer_encoder.layers[self.layers_info[0][0]].in_embeddings]

    def get_feature_encoder_predictions(self):
        return self.get_all_layers_predictions(embeddings=self.get_feature_encoder_embeddings())

    def get_all_layers_embeddings(self): #todo fix for n_estimators>1
        embeddings = []
        for l, info in self.layers_info:
            layer = self.predictor.model.transformer_encoder.layers[l]
            embeddings.append(layer.out_embeddings)
        return embeddings

    def get_decoder_pre_norm(self):
        return self.predictor.model.encoder_out_norm

    @torch.no_grad()
    def get_all_layers_predictions(self, embeddings=None, decoder_type = "default"): 
        if embeddings is None:
            embeddings = self.get_all_layers_embeddings()
        all_layer_prediction = []
        all_layer_predict_proba = []
        for l, emb in enumerate(embeddings):
            device = next(self.predictor.model.parameters()).device
            emb = self.predictor.model.encoder_out_norm(emb.to(device).float())
            
            if decoder_type ==  "default":
                decoder = self.get_decoder()
            if decoder_type ==  "finetuned":
                decoder = self.finetuned_decoders[l].to(device)
            
            decoder_outputs =decoder(emb[:, :, -1].transpose(0, 1).to(device).float()).transpose(0, 1)
            outputs: list[torch.Tensor] = []
            for id_pipe, output in enumerate(decoder_outputs):
                assert output.ndim == 2

                if self.predictor.softmax_temperature != 1:
                    output = (output[:, :self.predictor.n_classes].float() / self.predictor.softmax_temperature)

                output = output[..., self.predictor.class_permutations[id_pipe]]
                outputs.append(output)

            outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
            output = torch.stack(outputs).mean(dim=0)

            output = output.float().cpu().numpy()
            p = output / output.sum(axis=1, keepdims=True)

            all_layer_predict_proba.append(p)
            y = np.argmax(p, axis=-1)
            y = self.predictor.classes.take(np.asarray(y, dtype=np.intp))
            all_layer_prediction.append(y)
        return all_layer_prediction, all_layer_predict_proba

    def model_init(self):
        if self.predictor is None:
            self.predictor = LimiXPredictor(device=self.device, model_path=self.model_path, inference_config=self.inference_config, seed =self.random_state, layers_info=self.layers_info)
            self.predictor.model.to(device=self.device)
    def get_decoder(self):
        self.model_init()
        return self.predictor.model.cls_y_decoder

    def load_finetuned_decoders(self):
        self.model_init()
        self.finetuned_decoders =[]
        for l, info in self.layers_info:
            decoder = copy.deepcopy(self.get_decoder())
            model_state_dict = torch.load(self.finetuned_decoders_path  + f"/decoder_layer_{l}.pth", map_location=torch.device('cpu'))
            decoder.load_state_dict(model_state_dict)
            self.finetuned_decoders.append(decoder)
        return  self.finetuned_decoders

    def model_inference(self,  X, y , support_size):
        
        y_full = torch.cat([y, torch.zeros( (X.shape[0], X.shape[1] - support_size) , device=y.device, dtype=y.dtype)],dim=1).to(self.device)
        X_full = X.to(self.device)
        output = self.predictor.model(X_full, y_full,eval_pos=int(support_size), layers_info = self.layers_info)
        
        return output
