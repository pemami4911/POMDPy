from __future__ import absolute_import
import tensorflow as tf
import os
import abc
from future.utils import with_metaclass


class BaseTFSolver(with_metaclass(abc.ABCMeta)):
    """Abstract object representing an Reader model."""

    def __init__(self, agent, sess):
        self.model = agent.model
        self._saver = None
        self.sess = sess

        with tf.name_scope('experiment_summary'):
            with tf.name_scope('avg_undiscounted_return'):
                self.avg_undiscounted_return = tf.placeholder('float32', None)
                self.avg_undiscounted_return_std_dev = tf.placeholder('float32', None)
                self.undiscounted_mean_summary = tf.summary.scalar('mean', self.avg_undiscounted_return)
                self.undiscounted_std_dev_summary = tf.summary.scalar('std_dev', self.avg_undiscounted_return_std_dev)
            with tf.name_scope('avg_discounted_return'):
                self.avg_discounted_return = tf.placeholder('float32', None)
                self.avg_discounted_return_std_dev = tf.placeholder('float32', None)
                self.discounted_mean_summary = tf.summary.scalar('mean', self.avg_discounted_return)
                self.discounted_std_dev_summary = tf.summary.scalar('std_dev', self.avg_discounted_return_std_dev)

            self.experiment_summary = tf.summary.merge([self.undiscounted_mean_summary, self.undiscounted_std_dev_summary,
                                       self.discounted_mean_summary, self.discounted_std_dev_summary])

    @staticmethod
    @abc.abstractmethod
    def reset(agent, sess):
        """
        :param agent
        :param sess:
        :return:
        """

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")

        if not os.path.exists(self.model.ckpt_dir):
            os.makedirs(self.model.ckpt_dir)
        self.saver.save(self.sess, self.model.ckpt_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.model.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model.ckpt_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: {}".format(fname))
            return True
        else:
            print(" [!] Load FAILED: {}".format(self.model.ckpt_dir))
            return False

    # @property
    # def checkpoint_dir(self):
    #     return os.path.join('checkpoints', self.model_dir)
    #
    # @property
    # def model_dir(self):
    #     model_dir = self.model.env
    #     for k, v in self._attrs.items():
    #         if not k.startswith('_') and k not in ['display']:
    #             model_dir += "/%s-%s".format((k, ",".join([str(i) for i in v]))
    #             if type(v) == list else v)
    #     return model_dir + '/'

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
