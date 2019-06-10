import sys
import os
import bcolz
import ijson
import pickle
import numpy as np
import re
import unicodedata
from functools import reduce
from constants import *


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def preprocess_embedding():
    """
    Resources:
    https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """
    words = []
    index = 0
    word_to_index = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='%s/glove.dat' % EMBEDDING_DIR, mode='w')
    file_name = os.path.join(EMBEDDING_DIR, GLOVE_FILE)
    print("Start processing embeddings %s" % GLOVE_FILE)
    with open(file_name, "rb") as embedding_file:
        for line in embedding_file:
            split_line = line.split()
            word = split_line[0]
            words.append(word)
            word_to_index[word] = index
            index += 1
            embedding = np.array(split_line[1:]).astype(np.float)
            vectors.append(embedding)
            if index % 100000 == 0:
                say("processed {} embeddings\n".format(index))

    for word in [START_TOKEN, END_TOKEN, UNKNOWN_WORD]:
        words.append(word)
        word_to_index[word] = index
        index += 1
        vectors.append(np.random.normal(size=(EMBEDDING_DIM, )) * 10e-3)

    vectors = bcolz.carray(vectors[1:].reshape((int(len(vectors)/EMBEDDING_DIM), EMBEDDING_DIM)),
                           rootdir='%s/glove.dat' % EMBEDDING_DIR, mode='w')
    vectors.flush()
    pickle.dump(words, open(os.path.join(EMBEDDING_DIR, "glove_words.pkl"), 'wb'))
    pickle.dump(word_to_index, open(os.path.join(EMBEDDING_DIR, "glove_index.pkl"), 'wb'))
    print("Done processing embeddings %s" % GLOVE_FILE)


def get_paragraph_questions(json_file_name, vocab_file=None):
    """
    :param: json_file_name of the squad data set to parse
    :param: existing vocab to build on
    """
    json_file = open(json_file_name, "r")
    data = []
    if vocab_file is None:
        vocab = set()
    else:
        vocab_file = open(vocab_file, "rb")
        vocab = set(pickle.load(vocab_file))
        vocab_file.close()
    print("Start processing data set %s" % json_file_name)
    for item in ijson.items(json_file, "data.item"):
        for paragraphs in item["paragraphs"]:
            paragraph = paragraphs["context"]
            indices, sentences = extract_sentences(paragraph)
            for qa in paragraphs["qas"]:
                if not qa["is_impossible"]:
                    # add (sentence, question) pairs to data
                    if len(qa["answers"]) != 0:
                        answer = qa["answers"][0]
                        answer_start_index = int(answer["answer_start"])
                        sentence_index = 0
                        for i in range(len(indices)):
                            if answer_start_index > indices[i]:
                                sentence_index = i
                            else:
                                break
                        sentence = sentences[sentence_index]
                        data.append((normalize_string(sentence), normalize_string(qa["question"])))
                    # add words in question to vocabulary
                    for word in get_words(qa["question"]):
                        vocab.add(word)
            # add words in sentences to vocabulary
            for word in flatten(map(get_words, sentences)):
                vocab.add(word)
    vocab.add(START_TOKEN)
    vocab.add(END_TOKEN)
    vocab.add(UNKNOWN_WORD)
    pickle.dump(list(vocab), open(os.path.join(EMBEDDING_DIR, "vocab.pkl"), 'wb'))
    pickle.dump(data, open(os.path.join(EMBEDDING_DIR, "dev.pkl"), 'wb'))
    print("Done processing data set %s" % json_file_name)


def extract_sentences(paragraph):
    """
    :param paragraph: String
    :return: list of indices of each sentence's first character's index in the original paragraph, list of sentences
    """
    sentences = re.split(f'({re.escape(".")}|{re.escape("?")})', paragraph)
    # remove empty strings
    sentences = list(map(lambda x: x.strip(), filter(None, sentences)))

    # find indices of sentence in paragraph
    indices = []
    for sentence in sentences:
        if sentence not in [".", "?"]:
            index = paragraph.index(sentence)
            indices.append(index)

    # periods and question marks are items in the list, combine them with the sentences
    sentences = reduce(lambda x, y: x[:-1] + [x[-1] + y] if y in [".", "?"] else x + [y], sentences, [])

    return indices, sentences


def get_words(paragraph):
    # get normalized string, split by space, and remove <sos> and <eos>
    words = normalize_string(paragraph).split(" ")[1:-1]
    # remove empty strings
    return list(filter(None, words))


# Turn a Unicode string to plain ASCII (remove accents), thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-alphanumeric characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z0-9'&.!?]+", r" ", s)
    return "<sos> " + s.strip() + " <eos>"


def flatten(l):
    return [item for sublist in l for item in sublist]


if __name__ == "__main__":
    # preprocess_embedding()
    get_paragraph_questions(os.path.join(DATA_SET_DIR, DEV_FILE))
    get_paragraph_questions(os.path.join(DATA_SET_DIR, TRAIN_FILE), os.path.join(EMBEDDING_DIR, "vocab.pkl"))
