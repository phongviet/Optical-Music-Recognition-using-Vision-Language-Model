from torch import nn
from transformers import AutoModel, AutoTokenizer
from .Text_decoder_model import CustomTextDecoder, CustomTokenizer

class OMRModel(nn.Module):
    def __init__(self, vision_encoder_name, text_decoder_name, mlp_layers, decoder_config=None):
        """
        Initialize OMR Model

        :param vision_encoder_name: Name of the vision encoder model from HuggingFace
        :param text_decoder_name: Name of text decoder ("custom" for CustomTextDecoder)
        :param mlp_layers: List of hidden dimensions for MLP projection layers
        :param decoder_config: Dictionary with decoder configuration (required if text_decoder_name="custom")
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(vision_encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        if text_decoder_name == "custom":
            if decoder_config is None:
                raise ValueError("decoder_config is required when text_decoder_name='custom'")
            self.decoder = CustomTextDecoder(
                vocab_size=decoder_config['vocab_size'],
                d_model=decoder_config['d_model'],
                max_seq_len=decoder_config['max_seq_len'],
                n_layers=decoder_config['n_layers'],
                n_heads=decoder_config['n_heads'],
                emb_dim=decoder_config['emb_dim']
            )
            self.tokenizer = CustomTokenizer()

        self.encoder_dim = self.encoder.config.hidden_size
        print(f"Encoder hidden size: {self.encoder_dim}")

        self.decoder_dim = self.decoder.hidden_size
        print(f"Encoder hidden size: {self.decoder_dim}")

        layers = []
        input_dim = self.encoder_dim
        if mlp_layers is None:
            mlp_layers = []

        # Hidden layers
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(input_dim, self.decoder_dim))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(self.decoder_dim)


    def forward(self, x):

        return x