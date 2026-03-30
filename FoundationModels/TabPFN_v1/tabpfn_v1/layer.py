from functools import partial

from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, Dropout, LayerNorm
from torch.utils.checkpoint import checkpoint

import numpy as np

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, 
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn

        self.activation = _get_activation_fn(activation)
        
        self.attn_output_weights = None
        self.embeddings = None
        self.in_embeddings = None
        self.component_contribution_scores = {}
        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    @staticmethod
    def compute_contribution(out_np, src_np, single_eval_position: int):
        
        def norm_ratio_clipped(out_np, src_np, eps=1e-2, max_value=1e2):
            ratio = np.linalg.norm(out_np) / (np.linalg.norm(src_np) + eps)
            return min(ratio, max_value)

        def cosine_residual_similarity(a, b, axis=-1, eps=1e-8):
            r = a + b
            dot = np.sum(r * b, axis=axis)
            norm_a = np.linalg.norm(r, axis=axis)
            norm_b = np.linalg.norm(b, axis=axis)
            return dot / (norm_a * norm_b + eps)

        def cosine_similarity(a, b, axis=-1, eps=1e-8):
            dot = np.sum(a * b, axis=axis)
            norm_a = np.linalg.norm(a, axis=axis)
            norm_b = np.linalg.norm(b, axis=axis)
            return dot / (norm_a * norm_b + eps)

        contribution_scores = {}
        contribution_scores["norm"] = {}
        contribution_scores["norm"]["total"] = norm_ratio_clipped(out_np, src_np)
        contribution_scores["norm"]["support"] = norm_ratio_clipped(out_np[:single_eval_position], src_np[:single_eval_position])
        contribution_scores["norm"]["query"] = norm_ratio_clipped(out_np[single_eval_position:], src_np[single_eval_position:])

        contribution_scores["cosine"] = {}
        contribution_scores["cosine"]["total"] = float(cosine_similarity(out_np, src_np, axis=-1).mean())
        contribution_scores["cosine"]["support"] = float( cosine_similarity(out_np[:single_eval_position], src_np[:single_eval_position], axis=-1,).mean())
        contribution_scores["cosine"]["query"] = float(cosine_similarity(out_np[single_eval_position:], src_np[single_eval_position:], axis=-1, ).mean())
        contribution_scores["cosine_residual"] = {}
        contribution_scores["cosine_residual"]["total"] = float(cosine_residual_similarity(out_np, src_np, axis=-1).mean())
        contribution_scores["cosine_residual"]["support"] = float( cosine_residual_similarity(out_np[:single_eval_position], src_np[:single_eval_position], axis=-1,).mean())
        contribution_scores["cosine_residual"]["query"] = float(cosine_residual_similarity(out_np[single_eval_position:], src_np[single_eval_position:], axis=-1, ).mean())
        return contribution_scores


    def component_contribution(self, component_name, out, src, single_eval_position: int, task_info={} ):
        compute_component_contribution = task_info.get("compute_component_contribution", False)
        if compute_component_contribution:
            out_np = out.detach().cpu().numpy().squeeze()
            src_np = src.detach().cpu().numpy().squeeze()
            contribution_scores = self.compute_contribution(out_np, src_np, single_eval_position)
            self.component_contribution_scores[component_name] = contribution_scores


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, task_info = {}) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        self.attn_output_weights = None
        self.embeddings = None
        self.in_embeddings = src.clone().detach().cpu()

        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src
        if isinstance(src_mask, tuple):
            # global attention setup
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask

            num_global_tokens = global_src_mask.shape[0]
            num_train_tokens = trainset_src_mask.shape[0]

            global_tokens_src = src_[:num_global_tokens]
            train_tokens_src = src_[num_global_tokens:num_global_tokens+num_train_tokens]
            global_and_train_tokens_src = src_[:num_global_tokens+num_train_tokens]
            eval_tokens_src = src_[num_global_tokens+num_train_tokens:]


            attn = partial(checkpoint, self.self_attn) if self.recompute_attn else self.self_attn
            
            print("test the maps from encoder")
            # TODO: validate
            global_tokens_src2, global_attn_weights = attn(global_tokens_src, global_and_train_tokens_src, global_and_train_tokens_src, None, True, global_src_mask)
            train_tokens_src2, train_attn_weights = attn(train_tokens_src, global_tokens_src, global_tokens_src, None, True, trainset_src_mask)
            eval_tokens_src2, eval_attn_weights = attn(eval_tokens_src, src_, src_,
                                    None, True, valset_src_mask)

            src2 = torch.cat([global_tokens_src2, train_tokens_src2, eval_tokens_src2], dim=0)
            self.attn_output_weights  = torch.cat([global_attn_weights, train_attn_weights, eval_attn_weights], dim=1)

        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            # TODO: check the map acqusition
            src_left,  att_map_left = self.self_attn(src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position])
            src_right, att_map_right= self.self_attn(src_[single_eval_position:], src_[:single_eval_position], src_[:single_eval_position])
            
            src2 = torch.cat([src_left, src_right], dim=0)
            self.attn_output_weights = torch.cat([att_map_left, att_map_right], dim=1)
        else:
            if self.recompute_attn:
                # TODO: check the map acqusition
                src2, self.attn_output_weights = checkpoint(self.self_attn, src_, src_, src_, src_key_padding_mask, True, src_mask)
            else:
                src2, self.attn_output_weights = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        src2 = self.dropout1(src2)

        self.component_contribution("attn", src2, src, single_eval_position, task_info)

        src[:single_eval_position] = src[:single_eval_position] + src2[:single_eval_position] * task_info.get("w_attn_on_support", 1.0)
        src[single_eval_position:] = src[single_eval_position:] + src2[single_eval_position:] * task_info.get("w_attn_on_query", 1.0)

        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src2 = self.dropout2(src2)
        self.component_contribution("mlp", src2, src, single_eval_position, task_info)

        src[:single_eval_position] = src[:single_eval_position] + src2[:single_eval_position] * task_info.get("w_ffn_on_support", 1.0)
        src[single_eval_position:] = src[single_eval_position:] + src2[single_eval_position:] * task_info.get("w_ffn_on_query", 1.0)

        
        if not self.pre_norm:
            src = self.norm2(src)

        self.embeddings = src.clone().detach().cpu()
        in_embeddings = self.in_embeddings.clone()

        if not self.pre_norm:
            in_embeddings = self.norm1(in_embeddings.to(device=src.device))
            in_embeddings = self.norm2(in_embeddings).detach().cpu()

        self.component_contribution("layer", (self.embeddings - in_embeddings), in_embeddings, single_eval_position, task_info)

        return src
