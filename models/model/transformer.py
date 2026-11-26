from torch import nn

from models.model.decoder import Decoder
from models.model.landscape_encoder import LandscapeEncoder


class Transformer(nn.Module):
    """Decoder-only Transformer policy used by ALDes."""

    def __init__(
        self,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
        condition_on_features=False,
        conditioning_mode=None,
        feature_dim=None,
        context_hidden_dim=None,
        grammar=None,
    ):
        super().__init__()
        if not condition_on_features:
            conditioning_mode = "none"
        elif conditioning_mode is None:
            conditioning_mode = "film"
        feature_dim = d_model if feature_dim is None else feature_dim
        context_hidden_dim = (
            4 * d_model if context_hidden_dim is None else context_hidden_dim
        )
        self.conditioning_mode = conditioning_mode
        self.landscape_encoder = (
            LandscapeEncoder(feature_dim, d_model, context_hidden_dim)
            if conditioning_mode == "film"
            else None
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device,
            condition_on_features=condition_on_features,
            conditioning_mode=conditioning_mode,
            grammar=grammar,
        )

    def reset_conditioning_identity(self):
        for layer in self.decoder.layers:
            for film in (layer.attention_film, layer.ffn_film):
                if film is not None:
                    film.reset_identity()

    def forward(
        self,
        features,
        target,
        attention_positions,
        action=None,
        reference=False,
    ):
        # ``attention_positions`` is retained in the public call signature for
        # compatibility with the original ALDes training loop. The decoder
        # derives positional information internally.
        del attention_positions
        context = (
            self.landscape_encoder(features)
            if self.landscape_encoder is not None
            else features
        )
        return self.decoder(
            target,
            context,
            None,
            None,
            None,
            action,
            reference,
        )
