import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import math
import os

MAX_LEN = 64
MIN_LEN = 5
batch_size = 256
show = 5
epochs = 40
path = 'poetry.txt'
bast_model_path = './rnn_poetry_model.h5'
disabled_str = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
poetry = []
tokens = []
dicts = []
losses = []
dataset = None
model = None


class Dict:
    def __init__(self, token):
        self.size = len(token)
        # 字到编号
        self.word2id = {}
        # 编号到字
        self.id2word = {}
        for i, w in enumerate(token):
            self.id2word[i] = w
            self.word2id[w] = i
        self.cls = self.word2id['[CLS]']
        self.sep = self.word2id['[SEP]']
        self.unk = self.word2id['[UNK]']
        self.pad = self.word2id['[PAD]']

    def id_to_word(self, token_id):
        return self.id2word[token_id]

    def word_to_id(self, word):
        return self.word2id.get(word, self.unk)

    def encode(self, words):
        token_ids = [self.cls, ]
        for w in words:
            token_ids.append(self.word_to_id(w))
        token_ids.append(self.sep)
        return token_ids

    def decode(self, token_ids):
        words = []
        for i in token_ids:
            word = self.id_to_word(i)
            if word not in {'[CLS]', '[SEP]'}:
                words.append(word)
        return words


class PoetryDataSet:
    def __init__(self, data, batch_size, dicts):
        self.batch_size = batch_size
        self.data = data
        self.dicts = dicts
        self.steps = int(math.floor(len(self.data) / self.batch_size))

    def __len__(self):
        return self.steps

    def padding_line(self, length, line):
        padding = self.dicts.pad
        pad_len = length - len(line)
        if pad_len < 0:
            return line[:length]
        else:
            return line + [padding] * pad_len

    def __iter__(self):
        x = len(self.data)
        np.random.shuffle(self.data)
        for start in range(0, x, self.batch_size):
            end = min(start + self.batch_size, x)
            batch_data = []
            length = max(map(len, self.data[start:end]))
            for line in self.data[start:end]:
                encoded = self.dicts.encode(line)
                pad_line = self.padding_line(length + 2, encoded)
                batch_data.append(pad_line)

            batch_data = np.array(batch_data)
            yield batch_data[:, :-1], batch_data[:, 1:]

    def gen(self):
        while True:
            yield from self.__iter__()


def read_poetry():
    global path, disabled_str, poetry, MAX_LEN, MIN_LEN, tokens, dataset, dicts
    with open(path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for l in lines:
            if l.count(":") != 1:
                continue
            title, content = l.split(':', 1)
            if any(x in content for x in disabled_str):
                continue
            if len(content) < MIN_LEN or len(content) > MAX_LEN:
                continue
            poetry.append(content.replace('\n', ''))

    counter = collections.Counter()
    for i in poetry:
        counter.update(i)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [t for t, c in counter.items() if c >= MIN_LEN]
    dicts = Dict(tokens)
    dataset = PoetryDataSet(poetry, batch_size, dicts)


def draw():
    # 画loss曲线
    plt.plot(range(len(losses)), losses, 'r')
    plt.title('Training loss')
    plt.xlabel("batch")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.figure()
    plt.show()


def gen_poetry(text=''):
    global dicts, model
    token_ids = dicts.encode(text)
    token_ids = token_ids[:-1]
    while len(token_ids) < MAX_LEN:
        _probas = model.predict([token_ids, ])[0, -1, 3:]
        p_args = _probas.argsort()[::-1][:100]
        p = _probas[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        token_ids.append(target)
        if target == 3:
            break
    return ''.join(dicts.decode(token_ids))


class EveryEpoch(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss = float('inf')

    def on_epoch_end(self, epoch, logs):
        if logs['loss'] <= self.loss:
            self.loss = logs['loss']
            model.save(bast_model_path)
        print()
        for i in range(5):
            print(gen_poetry())

    def on_batch_end(self, batch, logs):
        losses.append(logs['loss'])


def load_rnn_model():
    global model
    model = tf.keras.models.load_model(bast_model_path)


if __name__ == '__main__':
    read_poetry()
    if os.path.isfile(bast_model_path):
        load_rnn_model()
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=dicts.size, output_dim=128),
            tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
            tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dicts.size, activation='softmax')),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy)
    # history = model.fit(dataset.gen(), steps_per_epoch=dataset.steps, epochs=epochs, callbacks=[EveryEpoch()])
    # draw()
