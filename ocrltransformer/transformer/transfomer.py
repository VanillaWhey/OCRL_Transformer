import torch
import torch.nn as nn

from .sub_layers import (
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
    VocabLogits,
    Embeddings,
)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, device="cpu"):
        super(Encoder, self).__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mask=False, device=device)
                for _ in range(num_blocks)
            ]
        )

        # self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        # x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, x, x, x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, device="cpu"):
        super(Decoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            mask=False,
            device=device,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mask=False, device=device)
                for _ in range(num_blocks)
            ]
        )

        self.vocab_logits = VocabLogits(embed_dim, vocab_size)

    def forward(self, encoder_outs, x):
        for block in self.transformer_blocks:
            output_seq_attention_out = self.multi_head_attention(
                query=x[:, -1:, :], key=x, value=x, residual_x=x[:, -1:, :]
            )
            x = block(
                query=output_seq_attention_out,
                key=encoder_outs,
                value=encoder_outs,
                residual_x=output_seq_attention_out,
            )
        return self.vocab_logits(x)


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_blocks,
        num_heads,
        encoder_vocab_size,
        output_vocab_size,
        device="cpu"
    ):
        super(TransformerTranslator, self).__init__()

        self.encoder_embedding = Embeddings(encoder_vocab_size, embed_dim, device=device)
        self.output_embedding = Embeddings(output_vocab_size, embed_dim, device=device)

        self.encoder = Encoder(embed_dim, num_heads, num_blocks, device=device)
        self.decoder = Decoder(
            embed_dim, num_heads, num_blocks, output_vocab_size, device=device
        )

        self.encoded = False

    def encode(self, input_sequence):
        embedding = self.encoder_embedding(input_sequence).to(self.device)
        self.encode_out = self.encoder(embedding)
        self.encoded = True

    def forward(self, output_sequence):
        if self.encoded == False:
            print("ERROR::TransformerTranslator:: MUST ENCODE FIRST.")
            return output_sequence
        else:
            embedding = self.output_embedding(output_sequence)
            return self.decoder(self.encode_out, embedding)
