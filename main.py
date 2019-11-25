import os
from model import SL

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 5, "The batch size used to train")
flags.DEFINE_integer('max_steps', 10000000, "Number of batches to run.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "trainlog", "Directory name to save the log")
flags.DEFINE_string("dataset", "image_train", "Directory name to load the data")
flags.DEFINE_string("datasetcontour", "contours_train", "Directory name to load the data")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean('log_device_placement', True, "Whether to log device placement.")
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    sl = SL(batch_size=FLAGS.batch_size)
    if FLAGS.is_train:
        sl.train()
    if FLAGS.visualize:
        sl.visualize()
if __name__ == '__main__':
    tf.app.run()
