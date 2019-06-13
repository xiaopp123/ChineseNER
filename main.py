import tensorflow as tf
import os
from collections import OrderedDict
import pickle
from loader import load_sentences, char_mapping, tag_mapping
from loader import prepare_data, BatchManager

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

def train():
    #load 
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)

#    update_tag_scheme(train_sentence, FLAGS.tag_schema)
    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            dico_chars_trian = char_mapping(train_sentences, FLAGS.lower)
            dico_chars, char_to_id, id_to_char = dico_chars_trian[0], dico_chars_trian[1], dico_chars_trian[2]
        else:
            dico_chars_trian = char_mapping(train_sentences, FLAGS.lower)
            dico_chars, char_to_id, id_to_char = dico_chars_trian[0], dico_chars_trian[1], dico_chars_trian[2]
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    print(tag_to_id)
    #[string, chars, segs, tags]
    train_data = prepare_data(train_sentences, char_to_id, tag_to_id, FLAGS.lower)

    train_manager = BatchManager(train_data, FLAGS.batch_size)

    config = config_model(char_to_id, tag_to_id)

#tf config
    tf_config = tf.ConfigProto()
    tf_config..gpu_options.allow_growth = True

    # num of batch
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = Model(config)

        sess.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            #read_values取出的值与不加一样
            emb_weight = sess.run(model.char_lookup.read_value())
            emb_weight = load_vec(config['emb_file'], id_to_char, config['char_dim'], emb_weights)



def main(_):
   if FLAGS.train:
       train()

if __name__ == '__main__':
    tf.app.run(main)
