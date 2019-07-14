import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict
import pickle
from loader import load_sentences, char_mapping, tag_mapping
from loader import prepare_data, BatchManager
from loader import load_word2vec, get_logger
from loader import test_ner, load_config
from loader import input_from_line
from model import Model

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_integer("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

FLAGS = tf.app.flags.FLAGS

def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower

    return config

def evaluate(sess, model, name, data, id_to_tag, logger):
    '''
    name: dev or trian
    '''
    logger.info("evaluate:{}".format(name))
    #返回真实结果[batch, ["char true_label pred_label"]]
    ner_results = model.evaluate(sess, data, id_to_tag)
    #print(ner_results) 
    #返回预测报告
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    print(type(eval_lines))

    for line in eval_lines:
        logger.info(line)

    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score: {:>.3f}".format(f1))

        return f1 > best_test_f1

    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score: {:>.3f}".format(f1))

        return f1 > best_test_f1

def train():
    #load data
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)

    #update_tag_scheme(train_sentence, FLAGS.tag_schema)
    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            #如果有预训练词表，获取训练集中字符的字符映射表
            #这里两种写成了一个
            dico_chars_trian = char_mapping(train_sentences, FLAGS.lower)
            dico_chars, char_to_id, id_to_char = dico_chars_trian[0], dico_chars_trian[1], dico_chars_trian[2]
        else:
            dico_chars_trian = char_mapping(train_sentences, FLAGS.lower)
            dico_chars, char_to_id, id_to_char = dico_chars_trian[0], dico_chars_trian[1], dico_chars_trian[2]

        #获取label映射表
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        #将字符映射表与label映射表保存
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        #如果已经保存就直接读取
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    print(tag_to_id)
    #[string, chars, segs, tags]
    #对原始数据进行编码
    train_data = prepare_data(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = prepare_data(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)

    #对数据划分batch
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, FLAGS.batch_size)

    #参数
    config = config_model(char_to_id, tag_to_id)

    #日志文件
    logger = get_logger(os.path.join("logs", FLAGS.log_file))

    #tf config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # num of batch
    steps_per_epoch = train_manager.len_data

    with tf.Session(config=tf_config) as sess:
        #创建模型
        model = Model(config)

        sess.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            #read_values取出的值与不加一样
            emb_weight = sess.run(model.char_lookup.read_value())
            emb_weight = load_word2vec(config['emb_file'], id_to_char, config['char_dim'], emb_weight)

        print("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(\
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)

            if best:
                #save_model()
                model.saver.save(sess, os.path.join(FLAGS.ckpt_path, "ner.ckpt"))
                logger.info("saved model")

def evaluate_line():
    """
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = Model(config)
        while True:
            line = input("请输入测试句子")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)
    """
    pass
    

def main(_):
    if FLAGS.train:
        train()
    else:
        evaluate_line()

if __name__ == '__main__':
    tf.app.run(main)
