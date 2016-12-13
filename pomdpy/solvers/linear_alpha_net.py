from __future__ import absolute_import
from pomdpy.util.ops import linear, select_action
from pomdpy.util.pickle_wrapper import save_pkl, load_pkl
from .alpha_vector import AlphaVector
from .base_tf_solver import BaseTFSolver
import tensorflow as tf
import numpy as np
import os


# TODO: Add a TF solver that takes the session as an argument
class LinearAlphaNet(BaseTFSolver):
    """
    Linear Alpha Network

    - linear FA for alpha vectors
    - 6 inputs (r(s,a))
    - 6 outputs (1 hyperplane per action)
    """

    def __init__(self, agent, sess):
        super(LinearAlphaNet, self).__init__(agent, sess)

        self.ops = {}
        self.w = {}
        self.summary_ops = {}
        self.summary_placeholders = {}
        self.w_input = {}
        self.w_assign_op = {}

        self.build_linear_network()

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        tf.global_variables_initializer().run()

    @staticmethod
    def reset(agent, sess):
        return LinearAlphaNet(agent, sess)

    def train(self):
        start_step = self.step_assign_op.eval({self.step_input: 0})

        total_reward, avg_reward_per_step, total_loss, total_v = 0., 0., 0., 0.
        avg_loss, avg_v = 0., 0.
        actions = []

        # Reset for new run
        belief = self.model.get_initial_belief_state()

        for step in range(start_step, self.model.max_steps):
            # 1. predict
            action, pred_v = self.predict(belief)
            # 2. act
            step_result = self.model.generate_step(action)

            if step_result.is_terminal:
                v_b_next = np.array([0.])
            else:
                next_belief = self.model.belief_update(belief, action, step_result.observation)
                # optionally clip reward
                # generate target
                _, v_b_next = self.predict(next_belief)

            target_v = self.model.discount * (step_result.reward + v_b_next)

            # compute gradient and do weight update
            _, loss = self.gradients(target_v, belief, step)

            total_loss += loss
            total_reward += step_result.reward
            total_v += pred_v[0]

            if step_result.is_terminal:
                # Reset for new run
                belief = self.model.get_initial_belief_state()

            actions.append(action)
            avg_reward_per_step = total_reward / (step + 1.)
            avg_loss = loss / (step + 1.)
            avg_v = total_v / (step + 1.)

            self.step_assign_op.eval({self.step_input: step + 1})

            self.inject_summary({
                'average.reward': avg_reward_per_step,
                'average.loss': avg_loss,
                'average.v': avg_v,
                'actions': actions,
                'training.learning_rate': self.ops['learning_rate_op'].eval(
                    {self.ops['learning_rate_step']: step + 1}),
            }, step + 1)

        print('\navg_reward: {}, avg_l: {}, avg_v: {}'.format(avg_reward_per_step, avg_loss, avg_v))

    def predict(self, belief):
        # try hard-coding input of linear net to be rewards (can try random as well)
        action, v_b, summary = self.sess.run([self.ops['a'], self.ops['v_b'], self.summary_ops['l1_out_summary']],
                                             feed_dict={
            self.ops['l0_in']: np.reshape(self.model.get_reward_matrix().flatten(), [1, 6]),
            self.ops['belief']: belief
        })

        # TODO: add step?
        self.summary_ops['writer'].add_summary(summary)

        return action, v_b

    def gradients(self, target_v, belief, learning_rate_step):
        return self.sess.run([self.ops['optim'], self.ops['loss']], feed_dict={
            self.ops['target_v']: target_v,
            self.ops['l0_in']: np.reshape(self.model.get_reward_matrix().flatten(), [1, 6]),
            self.ops['belief']: belief,
            self.ops['learning_rate_step']: learning_rate_step})

    def alpha_vectors(self):
        gamma = self.sess.run(self.ops['l1_out'], feed_dict={
            self.ops['l0_in']: np.reshape(self.model.get_reward_matrix().flatten(), [1, 6]),
            self.ops['belief']: self.model.get_initial_belief_state()  # unused
        })
        gamma = np.reshape(gamma, [self.model.num_actions, self.model.num_states])
        vector_set = set()
        for i in range(self.model.num_actions):
            vector_set.add(AlphaVector(a=i, v=gamma[i]))
        return vector_set

    def build_linear_network(self):
        with tf.variable_scope('linear_prediction'):
            l1_dim = 6
            # One linear layer
            self.ops['l0_in'] = tf.placeholder('float32', [1, self.model.num_states *
                                                           self.model.num_actions], name='linear_layer_input')
            self.ops['l1_out'], self.w['l1_w'], self.w['l1_b'] = \
                linear(self.ops['l0_in'], l1_dim, activation_fn=None, name='linear_layer')

            self.ops['belief'] = tf.placeholder('float32', [self.model.num_states], name='belief')

            self.ops['l1_out'] = tf.reshape(self.ops['l1_out'], [self.model.num_actions, self.model.num_states])

            vector_set = set()
            for i in range(self.model.num_actions):
                vector_set.add(AlphaVector(a=i, v=self.ops['l1_out'][i]))

            self.ops['a'], self.ops['v_b'] = select_action(self.ops['belief'], vector_set)

            aggregated_out = []
            for i in range(self.model.num_actions):
                aggregated_out.append(tf.histogram_summary('l1/{}'.format(i), self.ops['l1_out'][i]))

            self.summary_ops['l1_out_summary'] = tf.merge_summary(aggregated_out, 'hyperplane_values')

        with tf.variable_scope('linear_optimizer'):
            # MSE loss function
            self.ops['target_v'] = tf.placeholder('float32', [None], name='target_v')

            self.ops['delta'] = self.ops['target_v'] - self.ops['v_b']
            self.ops['clipped_delta'] = tf.clip_by_value(self.ops['delta'], -1, 1, name='clipped_delta')

            self.ops['loss'] = tf.reduce_mean(tf.square(self.ops['clipped_delta']), name='loss')
            self.ops['learning_rate_step'] = tf.placeholder('int64', None, name='learning_rate_step')
            self.ops['learning_rate_op'] = tf.maximum(self.model.learning_rate_minimum,
                                                      tf.train.exponential_decay(
                                                                 self.model.learning_rate,
                                                                 self.ops['learning_rate_step'],
                                                                 self.model.learning_rate_decay_step,
                                                                 self.model.learning_rate_decay,
                                                                 staircase=True))
            self.ops['optim'] = tf.train.GradientDescentOptimizer(self.ops['learning_rate_op'],
                                                                  name='GradientDescent'). \
                minimize(self.ops['loss'])

        with tf.variable_scope('linear_summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.v', 'training.learning_rate']

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops['{}'.format(tag)] = tf.scalar_summary('{}'.format(tag),
                                                                       self.summary_placeholders[tag])

            histogram_summary_tags = ['actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops['{}'.format(tag)] = tf.histogram_summary(tag, self.summary_placeholders[tag])

            self.summary_ops['writer'] = tf.train.SummaryWriter('./logs/', self.sess.graph)

        self.summary_ops['saver'] = tf.train.Saver(self.w, max_to_keep=30)

        self.load_model()

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops['{}'.format(tag)] for tag in tag_dict.keys()], feed_dict={
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
            })
        for summary_str in summary_str_lists:
            self.summary_ops['writer'].add_summary(summary_str, step)

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
