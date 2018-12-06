import sys
import os
import bcolz
import ijson
import pickle
import numpy as np
import re
import unicodedata

DATA_SET_DIR = "../data_set/squad"
TRAIN_FILE = "train-v2.0.json"
DEV_FILE = "dev-v2.0.json"
EMBEDDING_DIR = "../embedding"
GLOVE_FILE = "glove.840B.300d.txt"


def flatten(l):
    return [item for sublist in l for item in sublist]


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def preprocess_embedding():
    words = []
    index = 0
    word_to_index = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='%s/glove.dat' % EMBEDDING_DIR, mode='w')
    file_name = os.path.join(EMBEDDING_DIR, GLOVE_FILE)
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

    vectors = bcolz.carray(vectors[1:].reshape((int(len(vectors)/300), 300)), rootdir='%s/glove.dat' % EMBEDDING_DIR,
                           mode='w')
    vectors.flush()
    pickle.dump(words, open(os.path.join(EMBEDDING_DIR, "glove_words.pkl"), 'wb'))
    pickle.dump(word_to_index, open(os.path.join(EMBEDDING_DIR, "glove_index.pkl"), 'wb'))


def get_paragraph_questions(json_file_name):
    """
    :param: json_file_name of the squad data set to parse
    :return Dict[String: List[String]] dictionary where the key is the paragraph,
    and value is a list of questions asked from the paragraph
    """
    json_file = open(json_file_name, "r")
    mapping = {}
    vocab = set()
    for item in ijson.items(json_file, "data.item"):
        for paragraphs in item["paragraphs"]:
            paragraph = normalize_string(paragraphs["context"])
            questions = list(map(lambda x: normalize_string(x["question"]), paragraphs["qas"]))
            words = paragraph.split(" ") + flatten([q.split(" ") for q in questions])
            for word in words:
                vocab.add(word)
            mapping[paragraph] = questions
    pickle.dump(list(vocab), os.path.join(EMBEDDING_DIR, "vocab.pkl"), 'wb')
    pickle.dump(mapping, os.path.join(EMBEDDING_DIR, "paragraph_question.pkl"), 'wb')


# Turn a Unicode string to plain ASCII (remove accents), thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return "<sos> " + s + " <eos>"


if __name__ == "__main__":
    preprocess_embedding()
    get_paragraph_questions(os.path.join(DATA_SET_DIR, DEV_FILE))
