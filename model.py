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

        #set placeholder
        self.char_inputs = tf.placeholder(tf.int32, [None, None], name='char_inputs')
        self.seg_inputs = tf.placeholder(tf.int32, [None, None], name='seg_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='target')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, 1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        #embedding for chinese character and segmentation represention
        embedding = self.embedding_layer()

        #bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer() #[batch_size, num_steps, emb_size]

        #logits for tag
        self.logits = self.project_layer(lstm_outputs)

        #
        self.loss = self.loss_layer()

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
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_input))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                   self.seg_lookup = tf.get_variable(
                       name = "seg_embedding",
                       shape = [self.num_segs, slef.seg_dim],
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
                                        fw_lstm_cell, bw_lstm_cell, self.char_inputs, self.lengths)

        return tf.concat(outputs, axis=2)
    def project_layer(lstm_outputs):
        """
        lstm_outputs:[batch_size, num_steps, lstm_dim * 2]
        return [batch_size, num_steps, num_tags]
        """
        with tf.variable_scop("project"):
            with tf.variable_scope("hidden"):
                W1 = tf.get_variable("W1", shape=[self.lstm_dim * 2, self.lstm_dim],\
                                    dtype=tf.float32, initializer=self.initializer):
                b1 = tf.get_varibale("b1", shape=[self.lstm_dim],\
                        dtype=tf.flot32, initializer=tf.zeros_initializer())

                #reshape 
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])

                hidden = tf.tanh(tf.nn.xw_plus_b(ouput, W1, b1))

            #project to score of tags
            with tf.varibale_scope("logits"):
                W2 = tf.get_variable("W2", shape=[self.lstm_dim, self.num_tags],\
                                    dtype = tf.floate32, initializer = self.initializer)
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
        targets = tf.concat(
            [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
        )

        self.trans = tf.get_varaible(
            "transitions",
            shape = [self.num_tags + 1, self.num_tags + 1],
            initializer = self.initializer
        )
        log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
            inputs = logits
            tag_indices = targets,
            transition_params = self.trans,
            sequence_lengths = lengths + 1
        )

        return tf.reduce_mean(-log_likelihood)
