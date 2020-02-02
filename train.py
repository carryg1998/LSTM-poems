import tensorflow as tf
import numpy as np
import os
from model import lstm_model
from load_txt import load_txt, load_lrc, get_batches

batch_size = 50
learning_rate = 0.01
lstm_size = 128
num_layers = 2
checkpoint = "checkpoint/"
txt_path = "data/poems_res.txt"
model_prefix = "txt"
epochs = 50

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)

txt_res = load_txt(txt_path)
x_batches, y_batches = get_batches(batch_size, txt_res["encoded"])

model = lstm_model(batch_size=batch_size, lstm_size=lstm_size, num_layers=num_layers,
    learning_rate=learning_rate, num_classes=len(txt_res["vocabs"]))
model.gen_model(training=True)

saver = tf.train.Saver(tf.global_variables())
all_var = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(all_var)

    print("开始训练")

    n = len(txt_res["encoded"])//batch_size
    c = 0
    for e in range(epochs):
        for b in range(n):
            feed = {model.x_input:x_batches[b],model.y_input:y_batches[b]}
            loss, _, _ = sess.run([model.loss, model.final_state, model.train_optimizer],
                feed_dict=feed)
            print("epoch:%d batch:%d counter:%d loss:%.8f" % (e,b,c,loss))
            c += 1
        if e % 9 == 0:
            saver.save(sess, os.path.join(checkpoint, model_prefix), global_step=e)
