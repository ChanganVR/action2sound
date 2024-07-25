import torch.nn as nn
import torch


class Audio_Feat_Encoder_Posembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, input_shape, embed_dim, target_seq_len=16):
        super().__init__() 
        # a hack to do concat conditioning
        self.target_seq_len = target_seq_len
        self.ch, self.feat_len, self.orig_seq_len = input_shape
        if self.target_seq_len != -1:
            input_len = self.ch * self.feat_len
            self.linear = nn.Sequential(
                nn.Linear(input_len, 512),
                nn.GELU(),
                nn.Linear(512, embed_dim)
            )
            self.pos_emb = nn.Embedding(target_seq_len, embed_dim)
            self.target_seq_len = target_seq_len
            self.embed_dim = embed_dim

    def forward(self, x):
        assert x.shape[1:] == torch.Size([self.ch, self.feat_len, self.orig_seq_len]), f"x shape is wrong: {x.shape}, it should be {[-1, self.ch, self.feat_len, self.orig_seq_len]}"
        if self.target_seq_len == -1:
            return x
        
        x = x.reshape(-1, self.ch * self.feat_len, self.orig_seq_len)
        x = x.permute(0,2,1)
        x = self.linear(x)
        pos_embedding = self.pos_emb(torch.arange(self.target_seq_len, device=x.device).reshape(1,-1)).repeat(x.shape[0], 1, 1)
        x = x + pos_embedding
        return x

class Energy_Feat_Encoder_Posembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, input_len, embed_dim, target_seq_len=16):
        super().__init__() 
        self.pos_emb = nn.Embedding(target_seq_len, embed_dim)
        self.target_seq_len = target_seq_len
        self.embed_dim = embed_dim
        self.input_len = input_len

        # use a 1D conv to downsamlpe the input
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2),
            nn.GELU(),
            nn.Conv1d(32, 128, 3, 2),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, 3, 1),
        )

    def forward(self, x):
        assert x.shape[1:] == torch.Size([1, self.input_len]), f"x shape is wrong: {x.shape}, it should be {[-1, 1, self.input_len]}"
        x = self.conv(x).transpose(2,1)
        pos_embedding = self.pos_emb(torch.arange(self.target_seq_len, device=x.device).reshape(1,-1)).repeat(x.shape[0], 1, 1)
        x = x + pos_embedding
        return x
    
class Energy_Feat_Encoder_Posembed_MLP(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, input_len, embed_dim, target_seq_len=16):
        super().__init__() 
        self.target_seq_len = target_seq_len
        self.embed_dim = embed_dim
        self.input_len = input_len

        # use a 1D conv to downsamlpe the input
        self.linear = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        assert x.shape[1:] == torch.Size([1, self.input_len]), f"x shape is wrong: {x.shape}, it should be {[-1, 1, self.input_len]}"
        x = self.linear(x)
        return x

if __name__ == "__main__":
    input_len = 76
    target_seq_len = 9
    x = torch.randn(2, input_len)
    print(x.shape)
    model = Energy_Feat_Encoder_Posembed(input_len, 512, target_seq_len=target_seq_len)
    y = model(x)
    print(y.shape)
    print(y)
