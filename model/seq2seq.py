import io
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

path_vocabs = './vocabs'
path_data_input = './in.txt'
path_data_target = './out.txt'
test_text = '宝剑锋从磨砺出'
losses = []
accuracy = []
epochs = 5
BATCH_SIZE = 96
MAX_LEN = 64
units = 512
embedding_dim = 256


def draw():
    # 画accuracy曲线
    plt.plot(range(len(accuracy)), accuracy, 'b')
    plt.title('Training accuracy')
    plt.xlabel("batch per 100")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy"])
    plt.figure()
    plt.show()
    # 画loss曲线
    plt.plot(range(len(losses)), losses, 'r')
    plt.title('Training loss')
    plt.xlabel("batch per 100")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.figure()
    plt.show()


def max_length(tensor):
    return max(len(t) for t in tensor)


# 文件内容读取的工具类
class Reader:
    def __init__(self, input_file, target_file, vocab_file, batch_size):
        self.input_file = input_file
        self.target_file = target_file
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.vocabs = self.read_vocabs()
        self.vocabs_size = len(self.vocabs) + 1
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(self.vocabs):
            self.id2word[i] = w
            self.word2id[w] = i
        self.sta = self.word2id['<s>']
        self.end = self.word2id['</s>']
        self.inputs, self.targets = self.read_data()
        self.max_len_input, self.max_len_target = max_length(self.inputs), max_length(self.targets)
        self.input_train, self.input_val, self.target_train, self.target_val = train_test_split(self.inputs,
                                                                                                self.targets,
                                                                                                test_size=0.15)
        self.BUFFER_SIZE = len(self.input_train)
        self.steps_per_epoch = len(self.input_train) // self.batch_size
        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_train, self.target_train)).shuffle(
            self.BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_val, self.target_val)).shuffle(
            self.BUFFER_SIZE)
        self.val_dataset = self.val_dataset.batch(self.batch_size, drop_remainder=True)

    def read_vocabs(self):
        vocabs = ['<pad>']
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in lines:
                i = i[:-1]
                vocabs.append(i)
        return vocabs

    def id_to_word(self, token_id):
        return self.id2word[token_id]

    def word_to_id(self, word):
        return self.word2id.get(word)

    def encode(self, words):
        token_ids = [self.sta, ]
        for w in words:
            token_ids.append(self.word_to_id(w))
        token_ids.append(self.end)
        return token_ids

    def decode(self, token_ids):
        words = []
        for i in token_ids:
            word = self.id_to_word(i)
            if word not in {'<s>', '</s>'}:
                words.append(word)
        return words

    def read_data(self, nums=None, start=None):
        inputs = []
        targets = []
        input_lines = io.open(path_data_input, encoding='utf-8').read().strip().split('\n')
        target_lines = io.open(path_data_target, encoding='utf-8').read().strip().split('\n')
        for i in range(len(input_lines[start:nums])):
            input_words = [x for x in input_lines[i].split(' ') if x != '']
            target_words = [x for x in target_lines[i].split(' ') if x != '']
            if len(input_words) > MAX_LEN:
                input_words = input_words[:MAX_LEN - 1]
            if len(target_words) > MAX_LEN:
                target_words = target_words[:MAX_LEN - 1]
            target_words.append('</s>')
            input_words_ids = self.encode(input_words)
            target_words_ids = self.encode(target_words)
            inputs.append(input_words_ids)
            targets.append(target_words_ids)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets, padding='post')
        return inputs, targets


reader = Reader(path_data_input, path_data_target, path_vocabs, BATCH_SIZE)


# attention模型
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden)))
        attention_weight = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weight * values, axis=1)
        return context_vector, attention_weight


# 编码器模型
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


encoder = Encoder(reader.vocabs_size, embedding_dim, units, BATCH_SIZE)


# 解码器模型
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weight = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weight


decoder = Decoder(reader.vocabs_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


def train_step(inp, target, enc_hidden):
    loss = 0
    acc_meter = tf.keras.metrics.Accuracy()
    global encoder, decoder
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([reader.word_to_id('<s>')] * BATCH_SIZE, 1)
        for t in range(1, target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(target[:, t], predictions)
            predic = tf.math.argmax(predictions, axis=1)
            acc_meter.update_state(target[:, t], predic)
            dec_input = tf.expand_dims(target[:, t], 1)
    batch_loss = (loss / int(target.shape[1]))
    batch_acc = acc_meter.result()
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss, batch_acc


def training():
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        print(len(list(enumerate(reader.dataset.take(reader.steps_per_epoch)))))
        for (batch, (inp, targ)) in enumerate(reader.dataset.take(reader.steps_per_epoch)):
            batch_loss, batch_acc = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            print('Epoch {} Batch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1, batch, batch_loss.numpy(),
                                                                    batch_acc.numpy()))
            if batch % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                gen_down_couplet_for_test(test_text)
                losses.append(batch_loss)
                accuracy.append(batch_acc)
            if batch % 1000 == 0 and batch != 0:
                draw()
        # if (epoch + 1) % 2 == 0:
        # encoder.save(path_model_encoder_save)
        # decoder.save(path_model_decoder_save)
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / reader.steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        gen_down_couplet_for_test(test_text)
    draw()


def evaluate(text):
    global encoder, decoder
    text = reader.encode(text)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([text], maxlen=reader.max_len_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([reader.word_to_id('<s>')], 0)
    for i in range(reader.max_len_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        w = reader.id_to_word(predicted_id)
        if w == '</s>':
            return result, text
        if w not in {'<pad>'}:
            result += w
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, text


def load_seq2seq_model():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def gen_down_couplet_for_test(text):
    result, text = evaluate(text)
    print("Input : %s" % text)
    print("output: {}".format(result))


def gen_down_couplet(text):
    result, text = evaluate(text)
    return result


if __name__ == '__main__':
    load_seq2seq_model()
    training()
