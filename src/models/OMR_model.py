import torch
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
        # Load the full model first
        full_model = AutoModel.from_pretrained(vision_encoder_name)

        # Extract vision encoder for CLIP models, or use full model for others
        self.encoder = full_model.vision_model
        self.encoder_dim = full_model.config.vision_config.hidden_size

        # Freeze encoder parameters
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

    def forward(self, images, text_tokens=None, attention_mask=None, segment_overlap=0.25):
        """
        Process music sheet images with vertical segmentation and overlapping.

        Args:
            images: Input images [batch_size, channels, height, width]
            text_tokens: Target text tokens [batch_size, seq_len] (optional for training)
            attention_mask: Mask for text tokens [batch_size, seq_len] (optional)
            segment_overlap: Overlap ratio between segments (default: 0.25 for 1/4 overlap)

        Returns:
            logits: [batch_size, seq_len, vocab_size] for autoregressive generation
        """
        batch_size = images.shape[0]

        # 1. Segment images vertically with overlap
        image_segments = self._segment_images_with_overlap(images, overlap_ratio=segment_overlap)
        # Shape: [batch_size, num_segments, channels, segment_height, width]

        # 2. Process each segment through vision encoder
        all_visual_features = []
        num_segments = image_segments.shape[1]

        for seg_idx in range(num_segments):
            segment = image_segments[:, seg_idx]  # [batch_size, C, H, W]

            # Extract visual features from segment
            # Use interpolate_pos_encoding=True to handle non-standard image sizes
            encoder_output = self.encoder(pixel_values=segment, interpolate_pos_encoding=True)
            visual_features = encoder_output.last_hidden_state  # [B, num_patches, encoder_dim]

            # Project to decoder dimension
            visual_features = self.mlp(visual_features)  # [B, num_patches, decoder_dim]
            visual_features = self.norm(visual_features)

            all_visual_features.append(visual_features)

        # 3. Concatenate all segment features along sequence dimension
        # Shape: [batch_size, num_segments * num_patches, decoder_dim]
        combined_visual_features = torch.cat(all_visual_features, dim=1)

        # 4. Decode text with cross-attention to combined visual features
        if text_tokens is None:
            # Inference mode: generate tokens autoregressively
            return self.generate(combined_visual_features)

        # Training mode: teacher forcing with ground truth tokens
        logits = self.decoder(
            text=text_tokens,
            encoder_output=combined_visual_features,
            mask=attention_mask
        )

        return logits  # [batch_size, seq_len, vocab_size]


    def _segment_images_with_overlap(self, images, num_segments = 4, overlap_ratio=0.25):
        """
        Segment images vertically with specified overlap.

        Args:
            images: [batch_size, channels, height, width]
            overlap_ratio: Overlap between segments (0.25 = 1/4 overlap)

        Returns:
            segments: [batch_size, num_segments, channels, segment_height, width]
        """
        batch_size, channels, height, width = images.shape

        # Calculate segment height (you can make this configurable)
        segment_height = height // num_segments  # Divide into ~4 segments (adjust as needed)
        stride = int(segment_height * (1 - overlap_ratio))  # Stride with overlap

        segments = []
        start_idx = 0

        while start_idx + segment_height <= height:
            end_idx = start_idx + segment_height
            segment = images[:, :, start_idx:end_idx, :]  # [B, C, seg_H, W]
            segments.append(segment)
            start_idx += stride

        # Handle last segment if it doesn't fit perfectly
        if start_idx < height:
            segment = images[:, :, -segment_height:, :]  # Take last segment_height pixels
            segments.append(segment)

        # Stack segments: [batch_size, num_segments, channels, segment_height, width]
        return torch.stack(segments, dim=1)

    @torch.no_grad()
    def generate(self, visual_features, max_length=512, temperature=1.0):
        """
        Autoregressive generation from visual features.

        Args:
            visual_features: Encoded visual features [batch_size, src_seq_len, decoder_dim]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature

        Returns:
            generated_tokens: [batch_size, generated_length]
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device

        # Ensure max_length doesn't exceed decoder's max_seq_len
        max_length = min(max_length, self.decoder.max_seq_len)

        # Start with BOS token (assuming vocab index 0)
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.decoder(
                text=generated,
                encoder_output=visual_features,
                mask=None
            )

            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Sample next token (greedy decoding)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences generated EOS (assuming vocab index 1)
            if (next_token == 1).all():
                break

        return generated



