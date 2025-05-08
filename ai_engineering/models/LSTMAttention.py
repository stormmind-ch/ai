import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttention(nn.Module):
    """
    LSTM encoder + single-head additive or dot-product attention
    -----------------------------------------------------------
    Args
    ----
    input_size   : #features per timestep   (default: 5)
    hidden_size  : LSTM hidden width        (default: 128)
    num_layers   : stacked LSTM layers      (default: 2)
    bidirectional: use a Bi-LSTM encoder    (default: False)
    attn_type    : 'dot'  or  'additive'    (default: 'dot')
    out_features : #classes                 (default: 4)
    """
    def __init__(self,
                 input_size=5,
                 hidden_size=128,
                 num_layers=2,
                 bidirectional=False,
                 attn_type='additive',
                 out_features=4):
        super().__init__()

        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.attn_type = attn_type

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3
        )

        if attn_type == 'additive':
            # Bahdanau / additive attention parameters
            self.W_h = nn.Linear(self.directions * hidden_size, hidden_size, bias=False)
            self.W_s = nn.Linear(self.directions * hidden_size, hidden_size, bias=False)
            self.v   = nn.Linear(hidden_size, 1, bias=False)

        self.fc = nn.Linear(self.directions * hidden_size, out_features)

    # --------------------------------------------------------
    def forward(self, x):
        """
        x shape:  (batch, seq_len, input_size)
        """
        encoder_out, (h_n, c_n) = self.lstm(x)           # encoder_out: (B, T, D*H)

        # use the final hidden state from the *top* layer as the query
        query = h_n[-self.directions:].transpose(0, 1)   # (B, D, H)
        query = query.contiguous().view(x.size(0), -1)   # (B, D*H)

        if self.attn_type == 'dot':
            # ---------- dot-product attention ----------
            # scores = <encoder_out_t, query>
            scores = torch.bmm(encoder_out, query.unsqueeze(2)).squeeze(2)  # (B, T)
        else:
            # ---------- additive (Bahdanau) -------------
            # scores = v^T tanh(W_h h_t + W_s s)
            scores = self.v(torch.tanh(
                self.W_h(encoder_out) +                # (B, T, H)
                self.W_s(query.unsqueeze(1))           # (B, 1, H)
            )).squeeze(2)                              # (B, T)

        attn_weights = F.softmax(scores, dim=1)         # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1),  # (B, 1, T)
                            encoder_out).squeeze(1)     # (B, D*H)

        logits = self.fc(context)                       # (B, out_features)
        return logits
