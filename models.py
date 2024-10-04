import math

import torch
import torch.nn as nn

import utils


# =============================================================================
# CNN Models for Jigsaw Puzzle and Sort 4-digit MNIST
# =============================================================================


class UnscrambleMnistCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_pieces,
        image_size,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        out_dim,
    ):
        super().__init__()

        piece_size = image_size // num_pieces

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        piece_size_after_conv = (piece_size) // (2 * 1)
        dim_after_conv = piece_size_after_conv**2 * hidden_channels
        mlp_hid_dim = (dim_after_conv + out_dim) // 2

        self.mlp = nn.Sequential(
            nn.Linear(dim_after_conv, mlp_hid_dim), nn.ReLU(True), nn.Linear(mlp_hid_dim, out_dim)
        )

    def forward(self, batch_pieces):
        """
        Args:
            batch_pieces: shape [bs, num_pieces**2, c, h, w]

        Returns:
            shape [bs, num_pieces**2, out_dim]
        """
        batch_shape = batch_pieces.shape[:-3]
        batch_pieces = batch_pieces.flatten(end_dim=-4)

        conv_pieces = self.conv(batch_pieces)  # [bs*n, hidden_c, h, w]
        conv_pieces_flatten = conv_pieces.flatten(start_dim=-3)
        pieces_embd = self.mlp(conv_pieces_flatten)  # [bs*n, out_dim]
        pieces_embd = pieces_embd.unflatten(0, batch_shape)  # [bs, n, out_dim]

        return pieces_embd


class SortMnistCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_digits,
        image_size,
        hidden_channels1,
        kernel_size1,
        stride1,
        padding1,
        hidden_channels2,
        kernel_size2,
        stride2,
        padding2,
        out_dim,
    ):
        super(SortMnistCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels1, kernel_size1, stride1, padding1),
            nn.BatchNorm2d(hidden_channels1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size2, stride2, padding2),
            nn.BatchNorm2d(hidden_channels2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        piece_size_after_conv = (image_size) // (2 * 2)
        dim_after_conv = (piece_size_after_conv**2) * hidden_channels2 * num_digits

        self.mlp = nn.Linear(dim_after_conv, out_dim)

    def forward(self, pieces):
        """
        Args:
            pieces: shape [bs, num_pieces, c, h, w]

        Returns:
            shape [bs, num_pieces, out_dim]
        """
        batch_shape = pieces.shape[:-3]
        pieces = pieces.flatten(end_dim=-4)

        conv_pieces = self.conv(pieces)  # [bs*n, hidden_c, h, w]
        conv_pieces_flatten = conv_pieces.flatten(start_dim=-3)
        pieces_embd = self.mlp(conv_pieces_flatten)  # [bs*n, out_dim]
        pieces_embd = pieces_embd.unflatten(0, batch_shape)  # [bs, n, out_dim]

        return pieces_embd


# =============================================================================
# Scaler Embedding Models
# =============================================================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, time_mlp=True):
        super().__init__()
        self.time_mlp = time_mlp
        if time_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        if self.time_mlp:
            t_emb = self.mlp(t_emb)
        return t_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix with dimensions [max_len, d_model]
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # Add a dimension for batch size
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input embedding
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


class PlanePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            # eps = 1e-6
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2.0 * (torch.div(dim_t, 2, rounding_mode="trunc")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
        return pos


class ScalarEmbedding(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: shape [batch, m]
        Returns:
            shape [batch, m, d_model]
        """
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, device=x.device)  # [d_model]
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats
        )  # [d_model]

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [batch, m, d_model]
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return pos_x


# =============================================================================
# Transformer
# =============================================================================


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(CrossAttentionEncoderLayer, self).__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, edges, nodes):
        """
        Args:
            edges: [batch, n choose 2, d_model]
            nodes: [batch, n, d_model]
        """
        n = nodes.size(-2)
        attn_mask = utils.incidence_matrix_mask(n)
        cross_attn_out, _ = self.cross_attn(
            nodes, edges, edges, attn_mask=attn_mask
        )  # [batch, n, d_model]
        cross_attn_out = self.layer_norm1(cross_attn_out + nodes)

        ff_out = self.ff(cross_attn_out)
        out = self.layer_norm2(ff_out + cross_attn_out)

        return out


class EncoderLayers(nn.Module):
    def __init__(
        self,
        dataset,
        d_model,
        nhead,
        d_hid,
        nlayers,
        dropout,
        d_out_adjust,
        encoder,
    ):
        super(EncoderLayers, self).__init__()

        self.d_model = d_model
        self.d_out_adjust = d_out_adjust
        self.dataset = dataset
        self.nlayers = nlayers
        self.nhead = nhead
        self.d_hid = d_hid
        self.dropout = dropout
        self.encoder = encoder

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if dataset == "tsp" and d_out_adjust == "square" and self.encoder == "cross":
            assert nlayers % 2 == 0

        self.encoder_layers = nn.ModuleList([self.get_layer(i) for i in range(nlayers)])

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def get_layer(self, i):
        if i % 2 == 0 and self.encoder == "cross":
            return CrossAttentionEncoderLayer(self.d_model, self.nhead, self.d_hid, self.dropout)
        elif self.encoder == "original" or self.encoder == "cross":
            return nn.TransformerEncoderLayer(
                self.d_model, self.nhead, self.d_hid, self.dropout, batch_first=True
            )
        else:
            raise NotImplementedError

    def apply_layers_cross(self, embd, time_embd, edge_embd=None, attn_mask=None):
        assert self.dataset == "tsp"

        device = embd.device
        n = embd.size(-2)
        batch = embd.size(0)
        time_embd = time_embd.unsqueeze(-2)

        for i, layer in enumerate(self.encoder_layers):
            if i == self.nlayers - 1:  # last layer, self attention
                if self.d_out_adjust == "square":
                    pad = torch.zeros(batch, n, self.d_model, device=device)
                    embd = torch.cat([embd, pad], dim=-2)  # [NTb, 2n, d_model]
                    embd = self.pos_encoder(embd)
                embd = layer(embd + time_embd, src_mask=attn_mask)
            elif i % 2 == 0:  # cross attention
                embd = layer(edge_embd + time_embd, embd + time_embd)
            else:  # self attention
                embd = layer(embd + time_embd)

        return embd

    def apply_layers_self(self, embd, time_embd, attn_mask=None):
        time_embd = time_embd.unsqueeze(-2)
        embd = embd + time_embd
        for layer in self.encoder_layers:
            embd = layer(embd, src_mask=attn_mask)
        return embd

    def forward(self, embd, time_embd, edge_embd=None):
        device = embd.device
        n = embd.size(-2)
        batch = embd.size(0)

        # create attention mask
        attn_mask = None
        if self.d_out_adjust == "square":
            mask_up_right = torch.full((n, n), float("-inf"), device=device)
            mask_bottom_right = torch.triu(torch.full((n, n), float("-inf"), device=device))
            mask_right = torch.cat([mask_up_right, mask_bottom_right], dim=0)
            mask_left = torch.zeros((2 * n, n), device=device)
            attn_mask = torch.cat([mask_left, mask_right], dim=-1)

        # If not using cross attn, we pad the input to length 2n
        # If we are using cross attn, then we pad at the last encoder layer
        if self.d_out_adjust == "square" and (not self.encoder == "cross"):
            pad = torch.zeros(batch, n, self.d_model, device=device)
            embd = torch.cat([embd, pad], dim=-2)  # [NTb, 2n, d_model]

        embd = self.pos_encoder(embd)
        if edge_embd is not None:
            edge_embd = self.pos_encoder(edge_embd)

        if self.encoder == "cross":
            embd = self.apply_layers_cross(embd, time_embd, edge_embd, attn_mask)
        elif self.encoder == "original":
            embd = self.apply_layers_self(embd, time_embd, attn_mask)
        else:
            raise NotImplementedError

        out = self.norm(embd)
        return out


class ReverseDiffusion(nn.Module):
    """
    p_{theta}(x_{t-1} | x_t)
    """

    def __init__(
        self,
        dataset,
        in_channels,
        n,
        image_size,
        hidden_channels1,
        kernel_size1,
        stride1,
        padding1,
        hidden_channels2,
        kernel_size2,
        stride2,
        padding2,
        num_digits,
        d_model,
        nhead,
        d_hid,
        nlayers,
        dropout,
        d_out_adjust,
        use_pos_enc,
    ):
        super(ReverseDiffusion, self).__init__()

        self.d_model = d_model
        self.d_out_adjust = d_out_adjust
        self.attn_mask = None
        self.use_pos_enc = use_pos_enc
        self.dataset = dataset
        self.nlayers = nlayers

        self.encoder = "cross" if dataset == "tsp" else "original"
        if dataset == "tsp" and d_out_adjust == "square" and self.encoder == "cross":
            assert nlayers % 2 == 0

        self.time_embd = TimestepEmbedder(d_model, time_mlp=True)

        if dataset in ["unscramble-noisy-MNIST", "unscramble-MNIST", "unscramble-CIFAR10"]:
            self.pieces_embd = UnscrambleMnistCNN(
                in_channels,
                n,
                image_size,
                hidden_channels1,
                kernel_size1,
                stride1,
                padding1,
                d_model,
            )
        elif dataset == "sort-MNIST":
            self.pieces_embd = SortMnistCNN(
                in_channels,
                num_digits,
                image_size,
                hidden_channels1,
                kernel_size1,
                stride1,
                padding1,
                hidden_channels2,
                kernel_size2,
                stride2,
                padding2,
                d_model,
            )
        elif dataset == "tsp":
            self.pieces_embd = PlanePositionEmbedding(d_model // 2, normalize=True)
            if self.encoder == "cross":
                self.edge_dist_embd = ScalarEmbedding(d_model)
        else:
            raise NotImplementedError

        self.encoder_layers = EncoderLayers(
            dataset, d_model, nhead, d_hid, nlayers, dropout, d_out_adjust, self.encoder
        )

        if d_out_adjust == "0":
            self.dmodel_mlp = nn.Sequential(
                nn.Linear(d_model, d_hid), nn.ReLU(), nn.Linear(d_hid, 1)
            )

    def training_patch_embd(self, src, x_start):
        """
        Args:
            src: permutations, [N, T, b, n]
            x_start: [b, n, c, h, w]

        Returns:
            embedding of patches, shape [NTb, n, d]
        """
        edge_tokens = None
        if self.dataset == "tsp" and self.encoder == "cross":
            points = utils.permute_embd(src, x_start)  # [N, T, b, n, 2]
            edge_tokens = utils.points_to_pairwise_dist(points)  # [N, T, b, m]
            edge_tokens = self.edge_dist_embd(edge_tokens)  # [N, T, b, m, d_model]
            edge_tokens = torch.flatten(edge_tokens, end_dim=-3)  # [NTb, m, d_model]

        patch_embd = self.pieces_embd(x_start)  # [b, n, d_model]
        src = utils.permute_embd(src, patch_embd)  # [N, T, b, n, d_model]
        src = torch.flatten(src, end_dim=-3)  # [NTb, n, d_model]

        return src, edge_tokens

    def eval_patch_embd(self, src, x_original):
        """
        Args:
            src: permutations [batch, beam, n]
            x_original: [batch, n, c, h, w]

        Returns:
            embdedding of patches, shape [batch*beam, n, d]
        """
        edge_tokens = None
        if self.dataset == "tsp" and self.encoder == "cross":
            points = utils.permute_embd(src, x_original.unsqueeze(1))  # [batch, beam, n, 2]
            edge_tokens = utils.points_to_pairwise_dist(points)  # [batch, beam, m]
            edge_tokens = self.edge_dist_embd(edge_tokens)  # [batch, beam, m, d_model]
            edge_tokens = torch.flatten(edge_tokens, end_dim=-3)  # [batch*beam, m, d_model]

        patch_embd = self.pieces_embd(x_original)  # [batch, n, d_model]
        patch_embd = patch_embd.unsqueeze(-3)  # [batch, 1, n, d_model]
        src = utils.permute_embd(src, patch_embd)  # [batch, beam, n, d_model]
        src = torch.flatten(src, end_dim=-3)  # [batch*beam, n, d]

        return src, edge_tokens

    def forward(self, src, time, x_start):
        """
        Args:
            src: [N, T, b, n], or [b, beam, n]
            time: [N, T, b], or [b]
            x_start: [b, n, object_shape]

        Returns:
            logits of x_{t-1}
        """
        batch_shape = src.shape[:-1]
        n = src.size(-1)

        if self.training:
            time = time.expand(batch_shape)
            src, edge_tokens = self.training_patch_embd(src, x_start)
        else:
            time = time.unsqueeze(-1).expand(batch_shape)
            src, edge_tokens = self.eval_patch_embd(src, x_start)

        time = time.flatten()
        time = self.time_embd(time)  # [NTb, d]

        out = self.encoder_layers(src, time, edge_tokens)

        if self.d_out_adjust == "square":
            row, col = torch.split(out, [n, n], dim=-2)  # [NTb, n, d]
            combined_out = torch.matmul(row, col.transpose(-1, -2))  # [NTb, n, n]
            combined_out = combined_out.unflatten(0, batch_shape)
            return combined_out
        else:
            out = self.dmodel_mlp(out)  # [NTb, n(+-1), 1]
            out = out.unflatten(0, batch_shape).flatten(start_dim=-2)  # [N, T, b, n(+-1)]
            return out
