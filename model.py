from ResNET.model import *
from ops import get_soft_update_ops
import numpy as np


class DeepDeterministicPolicyGradient:

    def __init__(self, name, sess=None):
        self._name = name

        if sess is None:
            self.sess = tf.get_default_session()

        else:
            self.sess = sess

        self.view = None
        self.v = None
        self.d = None
        self.action = None
        self.training = None
        self.Q_real = None
        self.Q_pred = None
        self.p_pred = None
        self.action_grads = None
        self.action_grads_placeholder = None
        self.critic_loss = None
        self.actor_grads = None
        self.l_rate_Q = None
        self.l_rate_p = None
        self.train_op = None

    @property
    def name(self):
        return self._name

    def build(self, view_size):

        with tf.variable_scope(self.name):
            self.view = tf.placeholder(shape=(None, *view_size, 1), dtype=tf.float32)
            self.v = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            self.d = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            self.action = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            self.training = tf.placeholder(shape=(), dtype=tf.bool)
            self.Q_real = tf.placeholder(shape=(None, 1), dtype=tf.float32)

            with tf.variable_scope("critic"):
                fts_view_Q = create_resnet_18(self.view, 256, self.training, output_activation=tf.nn.elu)

                fts_v_Q = layers.dense(self.v, 64, tf.nn.elu)
                fts_v_Q = layers.dense(fts_v_Q, 128, tf.nn.elu)

                fts_d_Q = layers.dense(self.d, 64, tf.nn.elu)
                fts_d_Q = layers.dense(fts_d_Q, 128, tf.nn.elu)

                fts_Q = tf.concat([fts_view_Q, fts_v_Q, fts_d_Q], axis=-1)

                ent_fts = layers.dense(fts_Q, 512)
                ent_action = layers.dense(self.action, 512)
                ent_net = tf.nn.elu(ent_fts+ent_action)

                self.Q_pred = layers.dense(ent_net, 1)

            with tf.variable_scope("actor"):
                fts_view_p = create_resnet_18(self.view, 256, self.training, output_activation=tf.nn.elu)
                fts_view_p = tf.nn.elu(fts_view_p)

                fts_v_p = layers.dense(self.v, 64, tf.nn.elu)
                fts_v_p = layers.dense(fts_v_p, 128, tf.nn.elu)

                fts_d_p = layers.dense(self.d, 64, tf.nn.elu)
                fts_d_p = layers.dense(fts_d_p, 128, tf.nn.elu)

                fts_p = tf.concat([fts_view_p, fts_v_p, fts_d_p], axis=-1)

                self.p_pred = tf.clip_by_value(layers.dense(fts_p, 3), -1., 1.)

            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            critic_vars = [t_var for t_var in t_vars if "critic" in t_var.name]
            actor_vars = [t_var for t_var in t_vars if "actor" in t_var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
            critic_update_ops = [update_op for update_op in update_ops if "critic" in update_op.name]
            actor_update_ops = [update_op for update_op in update_ops if "actor" in update_op.name]

            self.critic_loss = tf.reduce_mean(tf.square(self.Q_pred - self.Q_real))

            self.action_grads = tf.gradients(self.Q_pred, self.action)[0]
            self.action_grads_placeholder = tf.placeholder(shape=(None, 3), dtype=tf.float32)       # for preventing entanglement between actor and critic
            self.actor_grads = tf.gradients(self.p_pred, actor_vars, -self.action_grads_placeholder)

            self.l_rate_Q = tf.placeholder(shape=(), dtype=tf.float32)
            self.l_rate_p = tf.placeholder(shape=(), dtype=tf.float32)

            with tf.control_dependencies(critic_update_ops):
                critic_train_op = tf.train.AdamOptimizer(self.l_rate_Q).minimize(self.critic_loss, var_list=critic_vars)

            with tf.control_dependencies(actor_update_ops):
                actor_train_op = tf.train.AdamOptimizer(self.l_rate_p).apply_gradients(zip(self.actor_grads, actor_vars))

            self.train_op = [critic_train_op, actor_train_op]

    def predict_Q(self, view, v, d, action):
        return self.sess.run(self.Q_pred, feed_dict={self.view: view, self.v: v, self.d: d, self.action: action, self.training: False})

    def predict_p(self, view, v, d):
        return self.sess.run(self.p_pred, feed_dict={self.view: view, self.v: v, self.d: d, self.training: False})

    def train(self, view, v, d, Q_real, action, l_rate_Q, l_rate_p):
        action_grads = self.sess.run(self.action_grads, feed_dict={self.view: view, self.v: v, self.d: d, self.action: action, self.training: False})

        critic_loss, _ = self.sess.run([self.critic_loss, self.train_op],
                                       feed_dict={self.view: view, self.v: v, self.d: d, self.Q_real: Q_real, self.action: action,
                                                  self.action_grads_placeholder: action_grads, self.l_rate_Q: l_rate_Q, self.l_rate_p: l_rate_p, self.training: True})

        return critic_loss

    def update(self, net, tau=0.1):
        self.sess.run(get_soft_update_ops(net, self.name, tau))





