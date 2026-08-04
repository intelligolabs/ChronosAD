import torch

import numpy as np
import torch.nn as nn

class GeneralLinearModel(nn.Module):
    def __init__(self,number_of_channels=2,hidden_size=256):
        super().__init__()
        self.number_of_channels = number_of_channels
        self.hidden_size = hidden_size

        self.__fcn1 = nn.Linear((self.hidden_size * number_of_channels) + number_of_channels, 256)
        self.__fcn2 = nn.Linear(256, 128)
        self.__fcn3 = nn.Linear(128, 64)
        self.__fcn4 = nn.Linear(64, 1)

        self.__BN1 = nn.BatchNorm1d(256)
        self.__BN2 = nn.BatchNorm1d(128)
        self.__BN3 = nn.BatchNorm1d(64)
    
    def forward(self, X):
        X = self.__fcn1(X)
        X = self.__BN1(X)
        X = nn.functional.selu(X)

        X = self.__fcn2(X)
        X = self.__BN2(X)
        X = nn.functional.selu(X)

        X = self.__fcn3(X)
        X = self.__BN3(X)
        X = nn.functional.selu(X)

        X = self.__fcn4(X)
        return X


class SWaTLinearModel(nn.Module):
    def __init__(self,number_of_channels=2,hidden_size=256):
        super().__init__()

        self.__fcn1 = nn.Linear((hidden_size * number_of_channels) + number_of_channels, 2048)
        self.__fcn2 = nn.Linear(2048, 1024)
        self.__fcn3 = nn.Linear(1024, 512)
        self.__fcn4 = nn.Linear(512, 128)
        self.__fcn5 = nn.Linear(128, 32)
        self.__fcn6 = nn.Linear(32, 1)

        self.__BN1 = nn.BatchNorm1d(2048)
        self.__BN2 = nn.BatchNorm1d(1024)
        self.__BN3 = nn.BatchNorm1d(512)
        self.__BN4 = nn.BatchNorm1d(128)
        self.__BN5 = nn.BatchNorm1d(32)
    
    def forward(self, X):
        X = self.__fcn1(X)
        X = self.__BN1(X)
        X = nn.functional.selu(X)

        X = self.__fcn2(X)
        X = self.__BN2(X)
        X = nn.functional.selu(X)

        X = self.__fcn3(X)
        X = self.__BN3(X)
        X = nn.functional.selu(X)

        X = self.__fcn4(X)
        X = self.__BN4(X)
        X = nn.functional.selu(X)

        X = self.__fcn5(X)
        X = self.__BN5(X)
        X = nn.functional.selu(X)

        X = self.__fcn6(X)
        return X

class ChronosLSTM(nn.Module):
    def __init__(self, device, number_of_channels=2, num_heads=4, hidden_size=256, embed_dim=1024, dataset=None):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.__lstms = nn.ModuleList()
        self.__attention = nn.ModuleList()
        self.__attention_nn = nn.ModuleList()

        for n in range(number_of_channels):
            self.__lstms.append(nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, num_layers=4, bidirectional=True))
            self.__attention.append(nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True))
            self.__attention_nn.append(nn.Linear(512,256))

        self.linear_model = None
        
        if dataset.lower() == 'swat':
            self.linear_model = SWaTLinearModel(number_of_channels, hidden_size)
        else:
            self.linear_model = GeneralLinearModel(number_of_channels, hidden_size)

    def normalizeBatch(self, data):
        embeddings = []
        scale = []

        for d in data:
            embeddings.append(d["embeddings"])
            scale.append(d["scale"])
        return embeddings, scale

    def forward(self, data):
        lstm_out_with_scale = []

        for idx, cols in enumerate(data.columns):
            embeddings, scale = self.normalizeBatch(data[cols].values)
            embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)
            embeddings = nn.functional.normalize(embeddings, dim=1)
            embeddings = nn.functional.selu(embeddings)
            scale = torch.tensor(np.array(scale), dtype=torch.float32, device=self.device)

            lstm_output, (hidden, cell_state) = self.__lstms[idx](embeddings)

            attn_output, attn_output_weights = self.__attention[idx](lstm_output, lstm_output, lstm_output)

            final_output = torch.einsum('bhs,bht->bt', attn_output_weights, attn_output)

            out = self.__attention_nn[idx](final_output)
            out = nn.functional.selu(out)

            out = torch.column_stack([out, scale])
            lstm_out_with_scale.append(out)

        X = torch.cat(lstm_out_with_scale, dim=1)

        X = self.linear_model(X)

        return X