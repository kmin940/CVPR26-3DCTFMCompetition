import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.0, ffn_mult=4):
        super().__init__()
        # Cross-attention: queries attend to keys/values from the image tokens
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)

        # Feed-forward network
        hidden = int(ffn_mult * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # K/V projection norm (optional but often helpful when only Q is learnable)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, query_qbd, kv_lbd):
        # query_qbd: [Q, B, D], kv_lbd: [L, B, D]
        # Pre-norm on K/V improves stability
        kv_lbd = self.norm_kv(kv_lbd)

        # Cross-attention
        attn_out, _ = self.mha(query=query_qbd, key=kv_lbd, value=kv_lbd)
        query_qbd = query_qbd + self.dropout_attn(attn_out)
        query_qbd = self.norm_attn(query_qbd)

        # Feed-forward
        ffn_out = self.ffn(query_qbd)
        query_qbd = query_qbd + self.dropout_ffn(ffn_out)
        query_qbd = self.norm_ffn(query_qbd)
        return query_qbd


class MultiLayersCrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0, num_layers=2, ffn_mult=1):
        super(MultiLayersCrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num

        # Learnable query vectors, shape [query_num, embed_dim]
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))

        # Stacked cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])

        # Final normalization and dropout
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_dropout = nn.Dropout(dropout)

        # Classifier maps [query_num * embed_dim] -> [num_classes]
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Args:
            x: image features of shape [B, D, H, W, L] or [B, D, H*W*L]
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]

        # Flatten spatial dimensions if needed: [B, D, H, W, L] -> [B, D, H*W*L]
        if x.dim() == 5:
            x = x.flatten(2)
        elif x.dim() == 6: # MIL case: [B, C, M, D, H, W] -> [B, C, M, D*H*W]
            x = x.flatten(2)

        # Convert to [L, B, D] for MultiheadAttention (seq_len, batch, embed_dim)
        kv_lbd = x.permute(2, 0, 1)

        # Expand learnable queries to batch: [Q, D] -> [Q, B, D]
        query_qbd = self.class_query.unsqueeze(1).expand(-1, B, -1)

        # Pass through stacked cross-attention blocks
        for layer in self.layers:
            query_qbd = layer(query_qbd, kv_lbd)

        # Final norm + dropout on queries
        query_qbd = self.final_norm(query_qbd)
        query_qbd = self.final_dropout(query_qbd)

        # [Q, B, D] -> [B, Q, D]
        queries_bqd = query_qbd.permute(1, 0, 2)

        # Concatenate all queries: [B, Q, D] -> [B, Q*D]
        queries_flat = queries_bqd.flatten(1)

        # Classify: [B, num_classes]
        logits = self.classifier(queries_flat)
        return logits


class MultiLayersCAImageQuery(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0, num_layers=2, ffn_mult=1):
        super(MultiLayersCAImageQuery, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = 1

        # Learnable query vectors, shape [query_num, embed_dim]
        #self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))

        # Stacked cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])

        # Final normalization and dropout
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_dropout = nn.Dropout(dropout)

        # Classifier maps [query_num * embed_dim] -> [num_classes]
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        #nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, img_query):
        """
        Args:
            x: image features of shape [B, D, H, W, L] or [B, D, H*W*L]
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]

        # Flatten spatial dimensions if needed: [B, D, H, W, L] -> [B, D, H*W*L]
        if x.dim() == 5:
            x = x.flatten(2)
        elif x.dim() == 6: # MIL case: [B, C, M, D, H, W] -> [B, C, M, D*H*W]
            x = x.flatten(2)

        # Convert to [L, B, D] for MultiheadAttention (seq_len, batch, embed_dim)
        kv_lbd = x.permute(2, 0, 1)

        # Expand learnable queries to batch: [Q, D] -> [Q, B, D]
        #query_qbd = self.class_query.unsqueeze(1).expand(-1, B, -1)
        query_qbd = img_query.unsqueeze(0) # [B, 1120] -> [1, B, 1120]

        # Pass through stacked cross-attention blocks
        for layer in self.layers:
            query_qbd = layer(query_qbd, kv_lbd)

        # Final norm + dropout on queries
        query_qbd = self.final_norm(query_qbd)
        query_qbd = self.final_dropout(query_qbd)

        # [Q, B, D] -> [B, Q, D]
        queries_bqd = query_qbd.permute(1, 0, 2)

        # Concatenate all queries: [B, Q, D] -> [B, Q*D]
        queries_flat = queries_bqd.flatten(1)

        # Classify: [B, num_classes]
        logits = self.classifier(queries_flat)
        return logits
