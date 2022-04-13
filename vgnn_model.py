import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadedGraphAttentionLayer, LayerNorm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class VGNN(nn.Module):
    def __init__(self, input_features, enc_features, dec_features, n_heads, n_layers,
                 dropout, alpha, variational=True, none_graph_features=0):
        super(VGNN, self).__init__()

        self.variational = variational  # whether the model is variational
        self.n_heads = n_heads  # number of heads for encoder and decoder self-attention layers
        self.n_layers = n_layers  # number of graph network layers
        self.enc_features = enc_features  # dimensionality of encoder
        self.dec_features = dec_features  # dimensionality of decoder
        # According to the original code, we should handle the first feature of eICU differently (previous readmission)
        # This variable allows us to exclude the first n features from graph layers.
        # It is processed by separately by a fully connected layer
        self.none_graph_features = none_graph_features  # number of first features to exclude from encoder/decoder
        self.input_features = input_features - none_graph_features

        # + 1 for the "new" node in the decoder (m = 1 in the paper).
        # This new node is fully connected with all encoder output nodes
        self.embed = nn.Embedding(self.input_features + 1, enc_features, padding_idx=0)

        # Multiple multi-headed self-attention encoder layers.
        self.encoder = [
            MultiHeadedGraphAttentionLayer(enc_features, enc_features, n_heads, dropout, alpha, 'encoder_attention_{}'.format(i), concat=True)
            for i in range(n_layers)
        ]
        for i, attention in enumerate(self.encoder):
            self.add_module('encoder_{}'.format(i), attention)

        # Single multi-headed self-attention decoder layer
        self.decoder = MultiHeadedGraphAttentionLayer(enc_features * n_heads, dec_features, n_heads, dropout, alpha,
                                                      "decoder_attention_1", concat=False)

        # final fully connected layer that returns a single prediction
        if self.none_graph_features > 0:
            # combine graph and non-graph features
            none_graph_hidden_features = dec_features // 2
            self.none_graph_layer = nn.Sequential(
                nn.Linear(none_graph_features, none_graph_hidden_features),
                nn.ReLU(),
                nn.Dropout(dropout))
            # combines results of graph and non-graph layers
            self.out_layer = nn.Sequential(
                nn.Linear(dec_features + none_graph_hidden_features, dec_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dec_features, 1))
        else:
            # all features are used in the graph layers
            self.out_layer = nn.Sequential(
                nn.Linear(dec_features, dec_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dec_features, 1))

        if self.variational:
            self.parameterize = nn.Linear(enc_features * n_heads, enc_features * n_heads * 2)
            self.dropout = nn.Dropout(dropout)


    # The intent of this function is to take a data sample (shape = (input_features)) and fully connect all existing nodes
    # input_edges - sparse representation of edges between existing nodes. Used in encoder
    # output_edges - sparse representation of edges between existing nodes + 1 new decoder node (m = 1 in paper). Used in decoder
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

    def reparametrize(self, mean, sigma):
        if self.training:
            # generate non-trainable random parameter epsilon from standard normal distribution
            eps = torch.randn_like(mean)
            return eps * sigma.exp() * 0.5 + mean
        else:
            return mean


    def forward(self, data):
        batch_decoded = []
        kld = []  # KL-divergence
        # sequential encode-decode logic for each item in the batch
        for i in range(data.shape[0]):
            graph_item = data[i, self.none_graph_features:]
            input_edges, output_edges = self.data_to_edges(graph_item)
            # Embed
            embedded = self.embed(torch.arange(self.input_features + 1).long().to(device))
            encoded = embedded[:-1]
            # Encode
            for encoder_layer in self.encoder:
                encoded = encoder_layer(encoded, input_edges)
                encoded = F.elu(encoded)

            if self.variational:
                parametrized = self.parameterize(encoded)
                parametrized = self.dropout(parametrized)
                mean = parametrized[:, :self.dec_features]
                sigma = parametrized[:, self.dec_features:]
                encoded = self.reparametrize(mean, sigma)
                mean = mean[graph_item == 1]
                sigma = sigma[graph_item == 1]
                kld.append(0.5 * torch.sum(sigma.exp() - sigma - 1 + mean.pow(2)) / mean.size()[0])

            # concat original nodes and the new decoder node representation
            encoded = torch.cat((encoded, embedded[-1].view(1, -1)))

            # Decode
            decoded = self.decoder(encoded, output_edges)
            decoded = F.relu(decoded)
            # leave only the last node, "decoder"
            decoded = decoded[-1]

            if self.none_graph_features > 0:
                # process non-graph features
                none_graph_item = data[i, :self.none_graph_features].type(torch.FloatTensor).to(device)
                decoded = torch.cat((decoded, self.none_graph_layer(none_graph_item)))

            batch_decoded.append(decoded)

        # Apply fully connected layers to the final node representation to get a single prediction
        prediction = self.out_layer(torch.stack(batch_decoded))
        total_kld = torch.sum(torch.stack(kld)) if self.variational else torch.tensor(0.0)
        return prediction, total_kld
    