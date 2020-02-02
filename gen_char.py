import tensorflow as tf
from model import lstm_model
from load_txt import load_txt, load_lrc, get_batches
import numpy as np

def get_word(pred, vocabs):
    pred = pred[0]       
    pred /= np.sum(pred)
    index = np.random.choice(np.arange(len(pred)), p=pred)
    return vocabs[index]

start_mark = 'B'
end_mark = 'E'
checkpoint = "checkpoint/"
txt_path = "data/poems_res.txt"
batch_size = 1
learning_rate = 0.01
lstm_size = 128
num_layers = 2

txt_res = load_txt(txt_path)

model = lstm_model(batch_size=batch_size, lstm_size=lstm_size,
    num_layers=num_layers, learning_rate=learning_rate, num_classes=len(txt_res["vocabs"]))
model.gen_model(training=False)

saver = tf.train.Saver(tf.global_variables())
all_var = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

first_word = input("输入第一个字：")

with tf.Session() as sess:
    sess.run(all_var)
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    saver.restore(sess, checkpoint)

    x = np.array([[txt_res["word_to_int"][start_mark]]])
    
    feed = {model.x_input:x}
    pred, last_state = sess.run([model.pred,model.final_state], feed_dict=feed)
    x = np.array([[txt_res["word_to_int"][first_word]]])
    word = first_word
    txt = ""

    c = 1
    while word != end_mark:
        txt = txt + word
        if c > 1000:
            break
        feed = {model.x_input:x, model.initial_state:last_state}
        pred, last_state = sess.run([model.pred,model.final_state], feed_dict=feed)
        word = get_word(pred, txt_res["vocabs"])
        x = np.array([[txt_res["word_to_int"][word]]])
        c += 1

print(txt)
