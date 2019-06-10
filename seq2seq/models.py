import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, inp, hidden):
        # embedded has size (seq_len x batch x hidden_size)
        embedded = self.embedding(inp).view(1, -1, self.embedding_dim)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        num_embeddings, embedding_dim = embedding_matrix.size()
        output_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        embedded = self.embedding(inp).view(1, -1, self.embedding_dim)
        output, hidden = self.gru(embedded, hidden)
        return self.softmax(self.out(output[0])), hidden
