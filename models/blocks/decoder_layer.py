from torch import nn

from models.layers.feature_wise_modulation import FeatureWiseModulation
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, context_dim=None):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.attention_film = (
            FeatureWiseModulation(context_dim, d_model)
            if context_dim is not None
            else None
        )
        self.ffn_film = (
            FeatureWiseModulation(context_dim, d_model)
            if context_dim is not None
            else None
        )

    def forward(self, dec, trg_mask, context=None):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if self.attention_film is not None:
            x = self.attention_film(x, context)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        if self.ffn_film is not None:
            x = self.ffn_film(x, context)

        return x
