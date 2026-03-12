"""
REINFORCE with Learned Baseline (Actor-Critic)
===============================================
The original algorithm from Nazari et al. (2018).

Loss:
    actor_loss  = mean( (R - V_stop_grad) × Σ log π(a|s) )
    critic_loss = MSE(R, V)

This is the simplest policy gradient method with a learned value function
as baseline to reduce variance.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rl_algorithms.base import BaseAlgorithm


class REINFORCE(BaseAlgorithm):

    @staticmethod
    def description():
        return ("REINFORCE with learned critic baseline (original paper). "
                "Simple policy gradient: (R - V) × log π.")

    @staticmethod
    def needs_critic():
        return True

    @staticmethod
    def build_train_step(agent, train_summary, args):
        R, v, logprobs, actions, idxs, batch, probs = train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)

        # Actor loss: REINFORCE with baseline
        actor_loss = tf.reduce_mean(tf.multiply((R - v_nograd), tf.add_n(logprobs)), 0)

        # Critic loss: MSE between predicted value and actual reward
        critic_loss = tf.losses.mean_squared_error(R, v)

        # Optimizers
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        # Compute and clip gradients
        actor_gra_and_var = actor_optim.compute_gradients(
            actor_loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'))
        critic_gra_and_var = critic_optim.compute_gradients(
            critic_loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'))

        clip_actor = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                      for grad, var in actor_gra_and_var]
        clip_critic = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                       for grad, var in critic_gra_and_var]

        actor_train_step = actor_optim.apply_gradients(clip_actor)
        critic_train_step = critic_optim.apply_gradients(clip_critic)

        return [actor_train_step, critic_train_step,
                actor_loss, critic_loss,
                actor_gra_and_var, critic_gra_and_var,
                R, v, logprobs, probs, actions, idxs]
