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

        my_dir = os.path.dirname(__file__)
        self.weight_dir = os.path.join(my_dir, '..', '..', 'experiments', 'pickle_jar')
        self.ckpt_dir = os.path.join(my_dir, '..', '..', 'experiments', 'checkpoints')
        self.tensorboard_log_dir = os.path.join(my_dir, '..', '..', 'experiments', 'tensorboard_logs')
        
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

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, self.ckpt_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.ckpt_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.ckpt_dir)
            return False

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
