import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=100,
        encoder_dropout=0.0,
        bidirectional=True,
        use_r_linear=True, #adds a MLP after RNN predictions
    ):
        super(RNNAutoEncoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = 1
        self.dropout = encoder_dropout
        self.bi = bidirectional
        self.use_r_linear = use_r_linear

        # for init
        self.first_dim = self.layers * 2 if self.bi else self.layers

        self.encoder_rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        self.reconstruct_rnn = nn.LSTM(
            self.input_size,
            1 if not self.use_r_linear else self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=False,
        )

        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

        fin_mid_size = int(fin_h_size / 2)

        self.reconstruct_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fin_h_size, fin_mid_size),
            nn.ReLU(),
            nn.Linear(fin_mid_size, 1),
        )

    def forward(self, inputs, input_lengths):
        packed_input = pack_padded_sequence(
            inputs, input_lengths, enforce_sorted=False, batch_first=True
        )

        self.batch_size = inputs.size()[0]

        _, (h_n, c_n) = self.encoder_rnn(packed_input)

        return h_n, c_n

    def reconstruct_decode(self, h_n, c_n, max_len):
        # forward HAS to be called before decode
        inp = self.dummy_decoder_input(batch_first=False)
        final_reconstructed = self.dummy_output(max_len, batch_first=False)
        for i in range(max_len):
            rnn_output, (h_n, c_n) = self.reconstruct_rnn(inp, (h_n, c_n))
            if self.use_r_linear:
                rnn_output = self.reconstruct_linear(rnn_output)

            final_reconstructed[i] = rnn_output

        return final_reconstructed.permute(1, 0, 2)

    def decode(self, **kwargs):

        # batch, 2, 100
        h_n = kwargs["r_h_n"]
        c_n = kwargs["r_c_n"]
        max_len = kwargs["r_max"]

        reconstructed = self.reconstruct_decode(h_n, c_n, max_len)

        return reconstructed

    def dummy_decoder_input(self, batch_first=True):
        # forward HAS to be called before decode
        if batch_first:
            dummy_inp = torch.zeros(self.batch_size, 1, self.input_size)
        else:
            dummy_inp = torch.zeros(1, self.batch_size, self.input_size)

        return dummy_inp.to(self.device())

    def dummy_output(self, max_len, batch_first=True):

        output_shape = 1
        # forward HAS to be called before decode
        if batch_first:
            dummy_out = torch.zeros(self.batch_size, max_len, output_shape)
        else:
            dummy_out = torch.zeros(max_len, self.batch_size, output_shape)

        return dummy_out.to(self.device())

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device