import numpy as np
import tensorflow as tf

class lstm_model:

    def __init__(self, batch_size, lstm_size, num_layers, learning_rate, num_classes):
        """
        初始化对象是传入神经网络参数
        """
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def gen_model(self, training=True):

        self.x_input = tf.placeholder(tf.int32, [self.batch_size, None])
        #如果不是训练，标签输入层设为None
        if training:
            self.y_input = tf.placeholder(tf.int32, [self.batch_size, None])
        else:
            self.y_input = None

        #定义LSTM层
        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        lstm = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        if training:
            self.initial_state = lstm.zero_state(batch_size=self.batch_size,dtype=tf.float32)
        else:
            self.initial_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        embedding = tf.get_variable('embedding', 
            initializer=tf.random_uniform([self.num_classes + 1, self.lstm_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, self.x_input)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(lstm, inputs, initial_state=self.initial_state)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.lstm_size])

        full_weight = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes + 1]))
        full_bias = tf.Variable(tf.zeros(shape=[self.num_classes + 1]))
        
        self.logits = tf.nn.bias_add(tf.matmul(lstm_outputs, full_weight), bias=full_bias)

        if training:
            labels = tf.one_hot(tf.reshape(self.y_input, [-1]), depth=self.num_classes + 1)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits))

            self.train_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        else:
            self.pred = tf.nn.softmax(self.logits)

