import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, LayerNorm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class VGNN(nn.Module):

    def __init__(self, input_features, enc_features, dec_features, n_heads, n_layers,
                 dropout, alpha, variational=True, none_graph_features=0):
        super(VGNN, self).__init__()

        self.variational = variational
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.input_features = input_features
        self.enc_features = enc_features
        self.dec_features = dec_features
        self.none_graph_features = none_graph_features

        # + 1 for the additional node in the decoder (m = 1 in the paper)
        self.input_features = input_features + 1 - none_graph_features

        self.embed = nn.Embedding(self.input_features, enc_features, padding_idx=0)

        # Single encoder layer attentions
        self.enc_att = [
            GraphAttentionLayer(enc_features, enc_features, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ]
        for i, attention in enumerate(self.enc_att):
            # TODO adjust name to handle multiple layers
            self.add_module('encoder_attention_{}'.format(i), attention)

        # Single decoder layer attentions
        self.dec_att = [
            GraphAttentionLayer(enc_features * n_heads, dec_features, dropout=dropout, alpha=alpha, concat=False)
            for _ in range(n_heads)
        ]
        for i, attention in enumerate(self.dec_att):
            # TODO adjust name to handle multiple layers
            self.add_module('decoder_attention_{}'.format(i), attention)

        # TODO handle the readmission feature

        self.dropout = nn.Dropout(dropout)

        # Layer normalization for encoder and decoder
        self.norm_enc = LayerNorm(enc_features * n_heads)
        self.norm_dec = LayerNorm(dec_features)

        # Linear combination of the decoded nodes (see forward for more details)
        self.V = nn.Linear(dec_features, dec_features)
        # final fully connected layer that returns prediction
        self.out_layer = nn.Sequential(
            nn.Linear(dec_features, dec_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_features, 1))

    # TODO REFACTOR THIS
    def data_to_edges(self, data):
        length = data.size()[0]
        nonzero = data.nonzero()
        if nonzero.size()[0] == 0:
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[length], [length]])
        if self.training:
            mask = torch.rand(nonzero.size()[0])
            mask = mask > 0.05
            nonzero = nonzero[mask]
            if nonzero.size()[0] == 0:
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[length], [length]])
        nonzero = nonzero.transpose(0, 1)
        lengths = nonzero.size()[1]
        input_edges = torch.cat((nonzero.repeat(1, lengths),
                                 nonzero.repeat(lengths, 1).transpose(0, 1)
                                 .contiguous().view((1, lengths ** 2))), dim=0)

        nonzero = torch.cat((nonzero, torch.LongTensor([[length]]).to(device)), dim=1)
        lengths = nonzero.size()[1]

        output_edges = torch.cat((nonzero.repeat(1, lengths),
                                  nonzero.repeat(lengths, 1).transpose(0, 1)
                                  .contiguous().view((1, lengths ** 2))), dim=0)
        return input_edges.to(device), output_edges.to(device)

        input_edges2 = torch.zeros(data.shape[0] + 1, data.shape[0] + 1)
        for i in range(input_edges.shape[1]):
            input_edges2[input_edges[0][i], input_edges[1][i]] = 1
        output_edges2 = torch.zeros(data.shape[0] + 1, data.shape[0] + 1)
        for i in range(output_edges.shape[1]):
            output_edges2[output_edges[0][i], output_edges[1][i]] = 1
        return input_edges2.to(device), output_edges2.to(device)

    def forward(self, data):
        batch_decoded = []
        # sequential encode-decode logic for each item in the batch
        for i in range(data.shape[0]):
            data_item = data[i, self.none_graph_features:]
            input_edges, output_edges = self.data_to_edges(data_item)
            # Embed
            embedded = self.embed(torch.arange(self.input_features).long().to(device))
            # Encode
            encoded = torch.cat([att(embedded, input_edges) for att in self.enc_att], dim=1)
            encoded = F.elu(self.norm_enc(encoded))
            # Decode
            decoded = torch.stack([att(encoded, output_edges) for att in self.dec_att], dim=0).mean(dim=0)
            # In my understanding, "V" combines all nodes together in the last "decoder" node.
            # Not 100% convinced about the need for it.
            decoded = self.V(F.relu(self.norm_dec(decoded)))
            # leave only the last node, "decoder"
            batch_decoded.append(decoded[-1])

        # Apply fully connected layers to the final node representation to get prediction
        prediction = self.out_layer(torch.stack([decoded for decoded in batch_decoded]))
        return prediction, torch.tensor(0.0)
