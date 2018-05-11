import tensorflow as tf
import numpy as np


def replay_DDPG(main, target, batch, gamma=0.85, l_rate_Q=0.001, l_rate_p=0.0001):
    view_stack = []
    v_stack = []
    d_stack = []
    Q_stack = []
    a_stack = []

    for state, action, reward, done, next_state in batch:
        view, v, d = state
        next_view_expanded = np.expand_dims(next_state[0], axis=0)
        next_v_expanded = np.expand_dims(next_state[1], axis=0)
        next_d_expanded = np.expand_dims(next_state[2], axis=0)

        if done:
            Q_real = reward

        else:
            Q_real = reward+gamma*target.predict_Q(next_view_expanded, next_v_expanded, next_d_expanded, target.predict_p(next_view_expanded, next_v_expanded, next_d_expanded))

        view_stack.append(view)
        v_stack.append(v)
        d_stack.append(d)
        Q_stack.append(Q_real)
        a_stack.append(action)

    view_stack = np.stack(view_stack, 0)
    v_stack = np.vstack(v_stack)
    d_stack = np.vstack(d_stack)
    Q_stack = np.vstack(Q_stack)
    a_stack = np.vstack(a_stack)

    return main.train(view_stack, v_stack, d_stack, Q_stack, a_stack, l_rate_Q, l_rate_p)


def get_soft_update_ops(net, target, tau):
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)

    update_ops = [target_var.assign(tau*net_var+(1-tau)*target_var) for net_var, target_var in zip(net_vars, target_vars)]

    return update_ops


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
