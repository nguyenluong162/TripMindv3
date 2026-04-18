import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1).unsqueeze(3)
        x = self.dropout(x)
        x = x.squeeze(3).permute(0, 2, 1)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out, mask):
        # lstm_out shape: [batch, seq_len, hidden_size]
        attn_weights = self.attention(lstm_out).squeeze(-1)
        # Bỏ qua các token PAD
        attn_weights = attn_weights.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Context vector: [batch, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

class TripMindEncoder(nn.Module):
    def __init__(self, vocab_size, num_categories=None, embed_dim=256, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.spatial_dropout = SpatialDropout1D(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Kích thước ẩn nhân 2 do BiLSTM (hai chiều)
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc_emb = nn.Linear(hidden_size * 2, embed_dim)
        
        self.num_categories = num_categories
        if num_categories is not None:
            self.category_classifier = nn.Linear(hidden_size * 2, num_categories)

    def forward(self, x):
        mask = (x == 0)
        
        x = self.embedding(x)
        x = self.spatial_dropout(x)
        
        lstm_out, _ = self.lstm(x)
        
        context = self.attention(lstm_out, mask)
        final_embedding = self.fc_emb(context)
        
        if self.num_categories is not None:
            cat_logits = self.category_classifier(context)
            return final_embedding, cat_logits
            
        return final_embedding