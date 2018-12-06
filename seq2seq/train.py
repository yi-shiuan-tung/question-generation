import torch
import torch.nn as nn
from torch import optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size