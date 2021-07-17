import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np

# class RNNAutoEncoder(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         hidden_size=100,
#         encoder_dropout=0.0,
#         bidirectional=True,
#         use_r_linear=True, #adds a MLP after RNN predictions
#     ):
#         super(RNNAutoEncoder, self).__init__()

#         self.input_size = input_dim
#         self.hidden_size = hidden_size
#         self.layers = 1
#         self.dropout = encoder_dropout
#         self.bi = bidirectional
#         self.use_r_linear = use_r_linear

#         # for init
#         self.first_dim = self.layers * 2 if self.bi else self.layers

#         self.encoder_rnn = nn.LSTM(
#             self.input_size,
#             self.hidden_size,
#             self.layers,
#             dropout=self.dropout,
#             bidirectional=self.bi,
#             batch_first=True,
#         )

#         self.reconstruct_rnn = nn.LSTM(
#             self.input_size,
#             1 if not self.use_r_linear else self.hidden_size,
#             self.layers,
#             dropout=self.dropout,
#             bidirectional=self.bi,
#             batch_first=False,
#         )

#         fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

#         fin_mid_size = int(fin_h_size / 2)

#         self.reconstruct_linear = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(fin_h_size, fin_mid_size),
#             nn.ReLU(),
#             nn.Linear(fin_mid_size, 1),
#         )

#     def forward(self, inputs, input_lengths):
#         packed_input = pack_padded_sequence(
#             inputs, input_lengths, enforce_sorted=False, batch_first=True
#         )

#         self.batch_size = inputs.size()[0]

#         encoder_outputs, (h_n, c_n) = self.encoder_rnn(packed_input)

#         return encoder_outputs, h_n, c_n

#     def reconstruct_decode(self, h_n, c_n, max_len):
#         # forward HAS to be called before decode
#         inp = self.dummy_decoder_input(batch_first=False)
#         final_reconstructed = self.dummy_output(max_len, batch_first=False)
#         for i in range(max_len):
#             rnn_output, (h_n, c_n) = self.reconstruct_rnn(inp, (h_n, c_n))
#             if self.use_r_linear:
#                 rnn_output = self.reconstruct_linear(rnn_output)

#             final_reconstructed[i] = rnn_output
#             inp = rnn_output

#         return final_reconstructed.permute(1, 0, 2)

#     def decode(self, **kwargs):

#         # batch, 2, 100
#         h_n = kwargs["r_h_n"]
#         c_n = kwargs["r_c_n"]
#         max_len = kwargs["r_max"]

#         reconstructed = self.reconstruct_decode(h_n, c_n, max_len)

#         return reconstructed

#     def dummy_decoder_input(self, batch_first=True):
#         # forward HAS to be called before decode
#         if batch_first:
#             dummy_inp = torch.zeros(self.batch_size, 1, self.input_size)
#         else:
#             dummy_inp = torch.zeros(1, self.batch_size, self.input_size)

#         return dummy_inp.to(self.device())

#     def dummy_output(self, max_len, batch_first=True):

#         output_shape = 1
#         # forward HAS to be called before decode
#         if batch_first:
#             dummy_out = torch.zeros(self.batch_size, max_len, output_shape)
#         else:
#             dummy_out = torch.zeros(max_len, self.batch_size, output_shape)

#         return dummy_out.to(self.device())

#     def device(self) -> torch.device:
#         """Heuristic to determine which device this module is on."""
#         first_param = next(self.parameters())
#         return first_param.device


class RNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=1,
        hidden_size=100,
        encoder_dropout=0.0,
        bidirectional=False,
    ):
        super(RNNEncoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = n_layers
        self.dropout = encoder_dropout
        self.bi = bidirectional

        self.encoder_rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

    def encode(self, inputs, input_lengths):
        packed_input = pack_padded_sequence(
            inputs, input_lengths, enforce_sorted=False, batch_first=True
        )

        self.batch_size = inputs.size()[0]
        max_len = int(torch.max(input_lengths).item())

        encoder_outputs, (h_n, c_n) = self.encoder_rnn(packed_input)
        encoder_outputs, _ = pad_packed_sequence(
            encoder_outputs, batch_first=True, total_length=max_len
        )
        # encoder_outputs -> [batch size, max seq lenght, hidden size]
        # h_n -> [1, batch size, hidden size]
        # c_n -> [1, batch size, hidden size]
        return encoder_outputs, h_n, c_n

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        max_seq_len = encoder_outputs.size(1)
        h = hidden.repeat(max_seq_len, 1, 1).transpose(0, 1)
        #h -> [batch size, max seq len, hidden size]
        #encoder_outputs -> [batch size, max seq len, hidden size]
        attn_energies = self.score(h, encoder_outputs)
        #attn_energies -> [batch size, max seq len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class RNNDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=1,
        hidden_size=100,
        dropout=0.0,
        bidirectional=False,
        use_r_linear=True,  # adds a MLP after RNN predictions
    ):
        super(RNNDecoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = n_layers
        self.dropout = dropout
        self.bi = bidirectional
        self.use_r_linear = use_r_linear

        self.reconstruct_rnn = nn.LSTM(
            self.input_size,
            1 if not self.use_r_linear else self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=False,
        )

        self.attention = Attention(hidden_size)

        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

        fin_mid_size = int(fin_h_size / 2)

        self.reconstruct_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fin_h_size, fin_mid_size),
            nn.ReLU(),
            nn.Linear(fin_mid_size, 1),
        )

    def reconstruct_decode(self, h_n, c_n, max_len):
        # forward HAS to be called before decode
        inp = self.dummy_decoder_input(batch_first=False)
        final_reconstructed = self.dummy_output(max_len, batch_first=False)
        for i in range(max_len):
            rnn_output, (h_n, c_n) = self.reconstruct_rnn(inp, (h_n, c_n))
            if self.use_r_linear:
                rnn_output = self.reconstruct_linear(rnn_output)

            final_reconstructed[i] = rnn_output
            inp = rnn_output

        return final_reconstructed.permute(1, 0, 2)

    def single_step_deocde(self, prev_decode_output, encoder_outputs, h_i, c_i):
        #prev_decoder_output -> [1, batch size, 1]

        attention_weights = self.attention(h_i, encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,1)
        context = context.transpose(0, 1)  # (1,B,1)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([prev_decode_output, context], 2)
        output, hidden = self.reconstruct_rnn(rnn_input, (h_i, c_i))
        if self.use_r_linear:
                output = self.reconstruct_linear(output)
        output = output.squeeze(0)  # (1,B,1) -> (B,1)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        return output, hidden, attention_weights


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


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(
            hidden_size + embed_size, hidden_size, n_layers, dropout=dropout
        )
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.size(1)
#         max_len = trg.size(0)
#         vocab_size = self.decoder.output_size
#         outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

#         encoder_output, hidden = self.encoder(src)
#         hidden = hidden[: self.decoder.n_layers]
#         output = Variable(trg.data[0, :])  # sos
#         for t in range(1, max_len):
#             output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
#             outputs[t] = output
#             is_teacher = random.random() < teacher_forcing_ratio
#             top1 = output.data.max(1)[1]
#             output = Variable(trg.data[t] if is_teacher else top1).cuda()
#         return outputs

if __name__ == "__main__":
    from dataloader import ReverseDataset
    from data_utils import pad_collate
    from torch.utils.data import DataLoader

    k = ReverseDataset()

    k_dataloader = DataLoader(
        k,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=pad_collate,
    )

    e = RNNEncoder(input_dim=1)
    a = Attention(hidden_size=100)

    for i, (x, y, lens) in enumerate(k_dataloader):
        out, h, c = e.encode(x, lens)
        print("encoder out", out.shape)
        print("encoder h", h.shape)
        print("encoder c", c.shape)

        attn = a(h, out)
        break