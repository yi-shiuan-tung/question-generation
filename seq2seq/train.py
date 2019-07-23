import os
import bcolz
import pickle
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.constants import *
from seq2seq.models import EncoderRNN, DecoderRNN
from seq2seq.plotting import show_plot
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import random
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5


def data_set_to_tensor(word_to_index, words, vocab_size):
    """
    Convert a list of words (String) to 1d long tensor of indices.
    Ex.
    word_to_index = {"hello": 0, "world": 1}
    vocab_size = 2
    words = ["hello", "world"]
    returns torch.tensor([0, 1])
    :param word_to_index Dict[String, Int]
    :param words: List[String]
    :param vocab_size
    :return:
    """
    tensor = torch.zeros(len(words), 1, device=device, dtype=torch.long)
    for i in range(len(words)):
        if words[i] not in word_to_index:
            tensor[i] = word_to_index[UNKNOWN_WORD]
        else:
            tensor[i] = word_to_index[words[i]]
    return tensor


def pad_tensor(tensor_list):
    """
    Given list of 2d tensors of varying sizes, add right padding such that every tensor has the same size in the 0
    dimension
    :param tensor_list: List[2d tensor]
    """
    max_tensor_length = max(map(lambda x: x.size()[0], tensor_list))
    for i in range(len(tensor_list)):
        tensor = tensor_list[i]
        if len(tensor.size()) != 2:
            raise Exception("Need 2d tensor as input")
        tensor_list[i] = F.pad(input=tensor,
                               pad=(0, 0, 0, max_tensor_length - tensor.size()[0]),
                               value=0)
    tensor_size = tensor_list[0].size()
    for tensor in tensor_list:
        if tensor.size() != tensor_size:
            raise Exception("Tensors not padded correctly, %s, %s" % (str(tensor.size()), str(tensor_size)))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train(train_data, word_to_index, vocabulary, encoder, decoder, n_iters, learning_rate=1e-3, batch_size=128):
    print("Starting training")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print_every = 10
    plot_every = 10

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    # def decay_function(episode):
    #     return learning_rate * (0.5 ** episode // 100)
    # encoder_scheduler = LambdaLR(encoder_optimizer, lr_lambda=decay_function)
    # decoder_scheduler = LambdaLR(decoder_optimizer, lr_lambda=decay_function)

    for epoch in range(1, n_iters + 1):
        input_tensors = []
        output_tensors = []
        for i in range(batch_size):
            sentence, question = random.choice(train_data)
            input_tensors.append(data_set_to_tensor(word_to_index, sentence.split(" "), len(vocabulary)))
            output_tensors.append(data_set_to_tensor(word_to_index, question.split(" "), len(vocabulary)))

        pad_tensor(input_tensors)
        pad_tensor(output_tensors)
        input_tensor = torch.stack(input_tensors)
        output_tensor = torch.stack(output_tensors)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        loss = run_epoch(input_tensor, output_tensor, word_to_index, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, criterion, batch_size)

        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # encoder_scheduler.step(epoch)
        # decoder_scheduler.step(epoch)

    show_plot(plot_losses)
    torch.save(encoder.state_dict(), "encoder.pkl")
    torch.save(decoder.state_dict(), "decoder.pkl")


def run_epoch(input_tensor, output_tensor, word_to_index, encoder, decoder, encoder_optimizer, decoder_optimizer,
              criterion, batch_size):
    encoder_hidden = encoder.init_hidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    output_length = output_tensor.size(1)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)

    # decoder input is a tensor of size (batch_size x 1) where the value is the index of the start token
    decoder_input = torch.tensor([[word_to_index[START_TOKEN]]] * batch_size, device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.float(), output_tensor[:, di, :].squeeze())
            decoder_input = output_tensor[:, di, :]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.float(), output_tensor[:, di, :].squeeze())
            decoder_input = decoder_output.argmax(dim=1)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length


def map_vocab_to_embedding(vocab):
    """

    :param vocab: set of words (string) that are from the data set
    :return:
        embedding_matrix: 2d tensor of size (vocab_size x embedding dim)
        vocab_to_index: Dict[String: Int] mapping from word to index in embedding_matrix
    """
    # get pre-trained embedding
    word_to_index = pickle.load(open(f'{EMBEDDING_DIR}/glove_index.pkl', 'rb'))
    words = pickle.load(open(f'{EMBEDDING_DIR}/glove_words.pkl', 'rb'))
    print("Number of words in embedding: %d" % len(words))

    vectors = bcolz.open(f'{EMBEDDING_DIR}/glove.dat')[:]
    glove = {w: vectors[word_to_index[w]] for w in words}

    vocab_to_index = {}
    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
    words_found = 0
    # go through vocab from data set, save embedding into matrix
    for i, word in enumerate(vocab):
        vocab_to_index[word] = i
        try:
            embedding_matrix[i] = glove[str.encode(word)]
            words_found += 1
        except KeyError:
            embedding_matrix[i] = glove[UNKNOWN_WORD]

    print("Number of words from data set that have embedding %d/%d" % (words_found, len(vocab)))
    return torch.tensor(embedding_matrix, device=device), vocab_to_index


def main():
    vocabulary = pickle.load(open(f'{EMBEDDING_DIR}/vocab.pkl', 'rb'))
    print("Number of words in data set: %d" % len(vocabulary))
    embedding_matrix, vocab_to_index = map_vocab_to_embedding(vocabulary)

    hidden_size = 600
    encoder = EncoderRNN(embedding_matrix, hidden_size)
    decoder = DecoderRNN(embedding_matrix, hidden_size)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    train_file = open(os.path.join(EMBEDDING_DIR, "train.pkl"), 'rb')
    train_data = pickle.load(train_file)
    train_file.close()
    n_iters = 2000
    train(train_data, vocab_to_index, vocabulary, encoder, decoder, n_iters)


if __name__ == "__main__":
    main()
