import os
from estimate import SL

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 25, "The batch size used to train")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to load the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "evallog", "Directory name to save the log")
flags.DEFINE_string("dataset", "test0603", "Directory name to load the data")
flags.DEFINE_string("datasetcontour", "test_contour0603", "Directory name to load the data")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean('log_device_placement', True, "Whether to log device placement.")
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        print('checkpoint dirictory not exist')
    sl = SL(batch_size=FLAGS.batch_size)
    sl.estimate()

if __name__ == '__main__':
    tf.app.run()
