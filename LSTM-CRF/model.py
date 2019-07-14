import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        #记录该模型在验证集和测试集最好的f1
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        #set placeholder
        self.char_inputs = tf.placeholder(tf.int32, [None, None], name='char_inputs')
        self.seg_inputs = tf.placeholder(tf.int32, [None, None], name='seg_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, 1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        #embedding for chinese character and segmentation represention
        self.input_embedding = self.embedding_layer()

        #bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer() #[batch_size, num_steps, emb_size]

        #logits for tag
        self.logits = self.project_layer(lstm_outputs)

        #
        self.loss = self.loss_layer()

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]\
                                  for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self):
        """
        """
        embedding = []
        with tf.variable_scope("char_embedding"):
            self.char_lookup = tf.get_variable(
                name = "char_embedding",
                shape = [self.num_chars, self.char_dim],
                initializer = self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, self.char_inputs))
            if self.seg_dim:
                with tf.variable_scope("seg_embedding"):
                   self.seg_lookup = tf.get_variable(
                       name = "seg_embedding",
                       shape = [self.num_segs, self.seg_dim],
                       initializer = self.initializer
                   )
                   embedding.append(tf.nn.embedding_lookup(self.seg_lookup, self.seg_inputs))

            embed = tf.concat(embedding, axis=-1)

        return embed

    def biLSTM_layer(self):
        """
        """
        with tf.variable_scope("char_BiLSTM"):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim)
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_dim)

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(\
                                        fw_lstm_cell, bw_lstm_cell, self.input_embedding, self.lengths, dtype=tf.float32)

        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs):
        """
        lstm_outputs:[batch_size, num_steps, lstm_dim * 2]
        return [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                W1 = tf.get_variable("W1", shape=[self.lstm_dim * 2, self.lstm_dim],\
                                    dtype=tf.float32, initializer=self.initializer)
                b1 = tf.get_variable("b1", shape=[self.lstm_dim],\
                        dtype=tf.float32, initializer=tf.zeros_initializer())

                #reshape 
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])

                hidden = tf.tanh(tf.nn.xw_plus_b(output, W1, b1))

            #project to score of tags
            with tf.variable_scope("logits"):
                W2 = tf.get_variable("W2", shape=[self.lstm_dim, self.num_tags],\
                                    dtype = tf.float32, initializer = self.initializer)
                b2 = tf.get_variable("b2", shape=[self.num_tags],\
                        dtype = tf.float32, initializer = tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W2, b2)

        return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self):
        """
        """
        small = -1000.0
        start_logits = tf.concat(
            [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],\
            axis = -1
        )
        pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
        logits = tf.concat([self.logits, pad_logits], axis=-1)
        logits = tf.concat([start_logits, logits], axis=1)

        #[batch_size, num_tags + 1]
        _targets = tf.concat(
            [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

        self.trans = tf.get_variable(
            "transitions",
            shape = [self.num_tags + 1, self.num_tags + 1],
            initializer = self.initializer
        )
        log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
            inputs = logits,
            tag_indices = _targets,
            transition_params = self.trans,
            sequence_lengths = self.lengths + 1
        )

        return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        is_train: flag train or test
        batch: train or test data
        return: dict feed
        """
        #batch是四部分组成
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.array(chars),
            self.seg_inputs: np.array(segs),
            self.dropout: 1.0, #这是保存多少，对于test全部保存
        }
        if is_train:
            feed_dict[self.targets] = np.array(tags, dtype="int32")
            feed_dict[self.dropout] = self.config["dropout_keep"]

        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        sess: session to run the batch
        is_train: weather it is a train batch
        batch: a dict containing batch data
        return: for training it returns steps, loss; for test return length, and logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            #训练的过程中，需要更改梯度，第三个即为更改梯度操作
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op], feed_dict
            )
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)

            return lengths, logits
    def decode(self, logits, lengths, matrix):
        """
        logits: [batch, num_steps, num_tags] float32
        lengths: [batch] int32, real length of each sequence
        matrix: transaction matrix for inference, [num_tags + 1, num_tags + 1]

        return:
        """
        paths = []
        small = -1000.0
        #start最后一个为什么是0
        #开始状态[1, num_tags + 1]
        start = np.array([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            #[length, num_tags + 1]
            logits = np.concatenate([score, pad], axis=1)
            #[length + 1, num_tags + 1]
            logits = np.concatenate([start, logits], axis=0)

            path, _ = tf.contrib.crf.viterbi_decode(logits, matrix)

            paths.append(path[1:])

        return paths

    def evaluate(self, sess, data_manager, id_to_tag):

        """
        sess:

        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [id_to_tag[x] for x in tags[i][:lengths[i]]]
                pred = [id_to_tag[x] for x in batch_paths[i][:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)

        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]

        return result_to_json(inputs[0][0], tags)
