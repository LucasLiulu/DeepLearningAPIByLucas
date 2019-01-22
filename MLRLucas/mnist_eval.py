import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from train_mnist import TrainModel

class MnistEval(object):
    def __init__(self):
        self.trainModel = TrainModel()
        self.EVAL_INTERVAL_SECS = 10

    def evaluate(self, mnist):
        x = tf.placeholder(tf.float32, [None, self.trainModel.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, self.trainModel.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        y = self.trainModel.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(self.trainModel.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        print(variables_to_restore)
        saver = tf.train.Saver(variables_to_restore)

        old_file_count = 0
        while True:
            if old_file_count > 5:
                break
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.trainModel.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    old_file_count = 0
                    model_ckpt_path = ckpt.model_checkpoint_path
                    saver.restore(sess, model_ckpt_path)
                    global_step = model_ckpt_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy: %g' % (global_step, accuracy_score))
                else:
                    old_file_count += 1
                    print("No checkpoint file found.")
                    return
            time.sleep(self.EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist_data_path = '.'
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
    MnistEval().evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
