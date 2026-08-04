import torch
from torch import nn
import numpy as np

class ChronosLSTM(nn.Module):
    def __init__(self, device, number_of_channels = 2):
        super().__init__()
        self.device = device
        
        self.__lstms = nn.ModuleList()
        
        for n in range(number_of_channels):
            self.__lstms.append(nn.LSTM(input_size=1024, hidden_size=256, batch_first=True, num_layers=4, bidirectional=True))
        
        self.__fcn1 = nn.Linear((256*number_of_channels) + number_of_channels, 256)
        self.__fcn2 = nn.Linear(256,128)
        self.__fcn3 = nn.Linear(128, 64)
        self.__fcn4 = nn.Linear(64,1)

        self.__BN1 = nn.BatchNorm1d(256)
        self.__BN2 = nn.BatchNorm1d(128)
        self.__BN3 = nn.BatchNorm1d(64)

    def normalizeBatch(self, data):
        embedings = []
        scale = []

        for d in data:
            embedings.append(d["embeddings"])
            scale.append(d["scale"])
        return embedings, scale
    def forward(self, data):
        scales = []
        lstm_out_with_scale = []
        for idx,cols in enumerate(data.columns):
            embedings, scale = self.normalizeBatch(data[cols].values)
            embedings = torch.tensor(np.array(embedings), dtype=torch.float32, device=self.device)
            embedings = nn.functional.normalize(embedings, dim=1)
            embedings = nn.functional.selu(embedings)
            scale = torch.tensor(np.array(scale), dtype=torch.float32, device=self.device)

            lstm_output, (hidden, cell_state) = self.__lstms[idx](embedings)
            h_n = hidden.view(4, 2, -1, 256)
            mean_direction = (h_n[-1, 0, :, :] + h_n[-1, 1, :, :]) / 2

            out = nn.functional.selu(mean_direction)

            out = torch.column_stack([out, scale])
            lstm_out_with_scale.append(out)

        X = torch.cat(lstm_out_with_scale, dim=1)

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