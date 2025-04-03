import os
import math
from typing import Optional, Union, List, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

from scaler import IOScaler
from simple_mlp import dump_json_file, get_prediction_input_data


def scaled_dot_product(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """scaled dot product function: softmax(Q @ K / sqrt(d)) @ V"""
    d_k = queries.size()[-1]
    attn_logits = torch.matmul(queries, keys.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, values)
    return values, attention


class MultiHeadAttention(nn.Module):
    """Multi Head Attention Layer"""

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Emebedding dim must be 0 modulo number of heads!")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], ret_attention: bool = False
    ):
        bs, traj_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(bs, traj_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(queries, keys, values, mask=mask)
        values = values.permute(0, 2, 1, 3)  # batch, traj_len, head, dims
        values = values.reshape(bs, traj_len, embed_dim)

        out = self.o_proj(values)
        if ret_attention:
            return out, attention
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.0
    ) -> None:
        """Encoder block with pre layer normalization"""
        super().__init__()

        # attention layer
        self.self_attention = MultiHeadAttention(input_dim, input_dim, num_heads)

        # 2-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        attn_out = self.self_attention(x_norm1, mask=mask)
        x = x + self.dropout(attn_out)

        # MLP part
        x_norm2 = self.norm2(x)
        linear_out = self.linear_net(x_norm2)
        x = x + self.dropout(linear_out)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerEncMLP(nn.Module):
    def __init__(self, cfg: dict, input_type: Union[str, List[str]], **kwargs) -> None:
        super().__init__()
        self.d_model = cfg["d_model"]
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        input_dims = cfg["n_features"]
        self.emb_net = nn.Sequential(
            nn.Linear(input_dims, self.d_model), nn.Dropout(cfg["dropout"])
        )
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg["num_layers"],
            input_dim=self.d_model,
            dim_feedforward=2 * self.d_model,
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
        self.decoder = None  # two stage approach
        self.out_dim = 2
        if "prediction_len" in cfg:
            intermediate_size = self.d_model // 2
            self.decoder = nn.Sequential(
                nn.Linear(
                    self.d_model,
                    intermediate_size,
                ),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(
                    intermediate_size,
                    self.out_dim * cfg["prediction_len"],
                ),
            )
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)

        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs_cat.size(0)
        x = self.emb_net(inputs_cat)
        x = self.positional_encoding(x)
        features = self.transformer_encoder(x, mask=mask)
        features = features[:, -1, :]
        out = None
        if self.decoder is not None:  # if None two stage approach
            out = self.decoder(features)
            out = out.view(bs, -1, self.out_dim)
        return out, features


class LightBaselineClassifier(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_name = kwargs["model_name"]
        data_cfg = kwargs["data_cfg"]
        network_cfg = kwargs["network_cfg"]
        hyperparameters_cfg = kwargs["hyperparameters_cfg"]

        saved_hyperparams = dict(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
        )

        self.save_hyperparameters(saved_hyperparams)

        self.metrics_per_class = {}
        self.sup_labels_mapping = data_cfg["supervised_labels"]
        self.n_labels = max(self.sup_labels_mapping.values())
        self.label_name = "data_label"
        if model_name == "transformer":
            first_dim = network_cfg["d_model"]
            self.encoder_tracklet = TransformerEncMLP(
                cfg=network_cfg,
                input_type=data_cfg["inputs"],
            )
        else:
            raise NotImplementedError(model_name)
        intermediate_size = first_dim // 2
        self.head_classifier = nn.Sequential(
            nn.Linear(
                first_dim,
                intermediate_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                self.n_labels + 1,
            ),
        )
        self.scaler = IOScaler(data_cfg["training_data_stats"])
        self.hyperparameters_cfg = hyperparameters_cfg
        self.loss = nn.CrossEntropyLoss()
        self.get_classification_data = partial(
            get_prediction_input_data,
            obs_len=data_cfg["obs_len"],
            inputs=data_cfg["inputs"],
        )

    def forward(self, x):
        obs_tracklet_data, y_gt = self.get_classification_data(x)
        scaled_train_batch = self.scaler.scale_inputs(obs_tracklet_data)
        _, predicted_features = self.encoder_tracklet(scaled_train_batch)
        y_hat = self.head_classifier(predicted_features)
        return y_hat

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=float(self.hyperparameters_cfg["lr"]),
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [opt], [
            dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        y_hat = self(train_batch)
        y_gt = train_batch[self.label_name][:, 0].long()
        loss = self.loss(y_hat, y_gt).mean()
        log_dict = dict(train_loss=loss)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int):
        y_hat = self(val_batch)
        y_gt = val_batch[self.label_name][:, 0].long()
        val_loss = self.loss(y_hat, y_gt).mean()
        log_dict = dict(val_loss=val_loss)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, on_step=False)
        preds = torch.nn.functional.softmax(y_hat, dim=1)
        self.update_metrics(preds, y_gt)

    def test_step(self, test_batch: dict, batch_idx: int):
        y_hat = self(test_batch)
        y_gt = test_batch[self.label_name][:, 0].long()
        preds = torch.nn.functional.softmax(y_hat, dim=1)
        self.update_metrics(preds, y_gt)
        self.update_metrics_per_class(preds, y_gt, test_batch[self.label_name][:, 0])

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            accuracy=Accuracy(
                task="multiclass", num_classes=self.n_labels + 1, average="micro"
            ).to(self.device),
            f1_score=F1Score(
                task="multiclass", num_classes=self.n_labels + 1, average="weighted"
            ).to(self.device),
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = dict(
            accuracy=Accuracy(
                task="multiclass", num_classes=self.n_labels + 1, average="micro"
            ).to(self.device),
            f1_score=F1Score(
                task="multiclass", num_classes=self.n_labels + 1, average="weighted"
            ).to(self.device),
        )
        for i in range(self.n_labels + 1):
            self.metrics_per_class[f"accuracy_c{i}"] = Accuracy(
                task="multiclass", num_classes=self.n_labels + 1, average="micro"
            ).to(self.device)
            self.metrics_per_class[f"f1_score_c{i}"] = F1Score(
                task="multiclass", num_classes=self.n_labels + 1, average="weighted"
            ).to(self.device)

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        eval_metrics = self.compute_metrics()
        if hasattr(self, "sup_labels_mapping"):
            eval_metrics.update(self.sup_labels_mapping)
        dump_json_file(eval_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> dict:
        model_out = self(predict_batch)
        preds = torch.nn.functional.softmax(model_out, dim=1)
        y_gt = predict_batch[self.label_name][:, 0]
        return dict(y_hat=preds, gt=y_gt)

    def update_metrics(self, y_hat: torch.Tensor, y_gt: torch.Tensor):
        for _, metric in self.eval_metrics.items():
            metric.update(preds=y_hat, target=y_gt)

    def update_metrics_per_class(
        self, y_hat: torch.Tensor, y_gt: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        for i, label in enumerate(labels):
            yh, gt = y_hat[i].unsqueeze(dim=0), y_gt[i].unsqueeze(dim=0)
            self.metrics_per_class[f"accuracy_c{int(label)}"].update(
                preds=yh, target=gt
            )
            self.metrics_per_class[f"f1_score_c{int(label)}"].update(
                preds=yh, target=gt
            )

    def compute_metrics(self) -> dict:
        final_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.eval_metrics.items()
        }
        final_clusters_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.metrics_per_class.items()
        }
        final_metrics.update(final_clusters_metrics)
        return final_metrics

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
        for _, metric in self.metrics_per_class.items():
            metric.reset()
