import tensorflow as tf
import os
from loader import load_sentences, char_mapping, tag_mapping

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
flags.DEFINE_float("batch_size",    20,         "batch size")
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



def main(_):
   if FLAGS.train:
       train()


if __name__ == '__main__':
    tf.app.run(main)
