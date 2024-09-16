# @Time : 2023/3/17 20:18
#  :LSM
# @FileName: RUN.py
# @Software: PyCharm
import torch
import torch.nn as nn

from layers.Embed import mark_Embedding, TokenEmbedding
from layers.Trend_fitting_block import RNN_layer
from layers.TRNet_EnDec import Hilly_Layer1, Corss_Attention_Layer, Hilly_Layer_cross, Hilly_Layer1_enc, CHAttn_Layer, Hilly_Layer, \
    ConvLayer, Tangle,  Hilly_Layer2
from layers.SelfAttention_Family import AttentionLayer, CrossHillyAttention, HillyAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.dropout = nn.Dropout(p=configs.dropout)
# A Recursive Layer
        self.A_Recursive_Layer_enc = nn.Sequential(
            RNN_layer(1, 1, self.pred_len, configs.dropout, select=0)
        )
        self.A_Recursive_Layer_dec = nn.Sequential(
            RNN_layer(1, 1, self.pred_len, configs.dropout, select=0)
        )
        # embedding mark
        self. mark_embedding = mark_Embedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                             configs.dropout)
        # embedding input
        self.input_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
 # CHAttn
        self.Residual_Component = CHAttn_Layer(
            [
                Hilly_Layer(
                    AttentionLayer(
                        HillyAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
# HAtnn-AR
        self.trend_embedding1 = TokenEmbedding(configs.enc_in, configs.d_model)
        self.A_Hilly_Attention_Layer1 = Hilly_Layer1(
            [
                Hilly_Layer1_enc(
                    AttentionLayer(
                        HillyAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.trend_embedding2 = TokenEmbedding(configs.enc_in, configs.d_model)
        self.A_Hilly_Attention_Layer2 = Hilly_Layer2(
            [
                Hilly_Layer_cross(
                    AttentionLayer(
                        CrossHillyAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=None
        )
        self.trend_embedding2 = TokenEmbedding(configs.enc_in, configs.d_model)
        self.A_Corss_Attention_Layer = Corss_Attention_Layer(
            [
                Hilly_Layer_cross(
                    AttentionLayer(
                        CrossHillyAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=None
        )

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.entangle = Tangle(configs.d_model, configs.dropout)
        self.linear = nn.Linear(configs.d_model, out_features=1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_dec = x_dec[:, -self.label_len - self.pred_len:-self.pred_len, :]

# A Recursive Layer
        input_padding, trend_input = self.A_Recursive_Layer_enc(x_enc)
        output_padding, trend_output = self.A_Recursive_Layer_dec(x_dec)

# CHAttn
        enc_error = (x_enc - trend_input).to(device=x_dec.device).float()
        dec_error = (x_dec - trend_output).to(device=x_dec.device).float()
        dec_error_with_zero = torch.cat(
            [dec_error, torch.zeros_like(dec_error)[:, :self.pred_len, :].to(device=x_enc.device)],
            dim=1)
        # Embedding
        embedding_encoder = torch.ones_like(x_enc)
        embedding_encoder = self.mark_embedding(embedding_encoder, x_mark_enc[:, -self.seq_len:, :]).cuda()
        embedding_decoder = torch.ones_like(x_dec)
        embedding_decoder = self.mark_embedding(embedding_decoder, x_mark_dec[:,  -self.label_len:, :]).cuda()

        enc_error_emb = self.dropout(self.input_embedding(enc_error)+embedding_encoder)
        dec_error_emb = self.dropout(self.input_embedding(dec_error_with_zero[:, -self.label_len:, :])+embedding_decoder)
        Xt = self.A_Corss_Attention_Layer(dec_error_emb, enc_error_emb, x_mask=None, cross_mask=None)

 # HAttn-AR
        ## A Hilly Attention Layer 1
        self_attention_fitting_vector = torch.cat(
            [x_enc, torch.zeros_like(x_enc)[:, -self.pred_len:, :].to(device=x_enc.device)], dim=1)

        self_attention_fitting_features = self.dropout(self.input_embedding(self_attention_fitting_vector[:, -self.seq_len:,:])+embedding_encoder).to( device=x_enc.device)

        self_attention_fitting_features, _ = self.A_Hilly_Attention_Layer1(self_attention_fitting_features,
                                                                           attn_mask=None)
        ## A Hilly Attention Layer 2
        TRNet_fitting_vector = torch.cat([x_enc, input_padding[:, :self.pred_len, :]], dim=1).float()

        TRNet_fitting_features = self.dropout(self.input_embedding(TRNet_fitting_vector[:, -self.seq_len:,:])+embedding_encoder).to( device=x_enc.device)

        Xtao= self.A_Hilly_Attention_Layer2(self_attention_fitting_features, TRNet_fitting_features)

        output_padding = output_padding.to(device=x_dec.device).float()

        trend_output_fitting_features = self.dropout(self.input_embedding(output_padding[:, -self.label_len:,:])+embedding_decoder)
        ## A Hilly Attention Layer 3
        Xs = self.A_Corss_Attention_Layer(trend_output_fitting_features, Xtao, x_mask=None, cross_mask=None)
# Entangle Component
        output = self.entangle(Xs, Xt)
        return self.linear(output)

    def forward(self, train_data, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        outputs = dec_out[:, -self.pred_len:, :]
        x_dec = x_dec[:, -self.pred_len:, :]
        outputs = outputs.reshape(outputs.size(1), -1)
        batch_y = x_dec.reshape(x_dec.size(1), -1)
        outputs = train_data.scaler.inverse_transform(outputs)
        batch_y = train_data.scaler.inverse_transform(batch_y)
        return outputs, batch_y
