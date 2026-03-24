"""
A2C — Advantage Actor-Critic
=============================
Extends REINFORCE by adding an entropy bonus to the actor loss.

Loss:
    advantage   = R - V_stop_grad
    actor_loss  = mean( advantage × Σ log π ) - entropy_coeff × H(π)
    critic_loss = MSE(R, V)

The entropy term H(π) encourages the policy to stay exploratory, preventing
premature convergence to suboptimal routes. This is especially useful for
routing problems where there are many local optima.

Reference:
    Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning."
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .base import BaseAlgorithm


class A2C(BaseAlgorithm):

    @staticmethod
    def description():
        return ("A2C (Advantage Actor-Critic). Adds entropy bonus to REINFORCE: "
                "advantage × log π - entropy_coeff × H(π). "
                "Encourages exploration, avoids local optima.")

    @staticmethod
    def needs_critic():
        return True

    @staticmethod
    def build_train_step(agent, train_summary, args):
        R, v, logprobs, actions, idxs, batch, probs = train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)
        advantage = R - v_nograd

        entropy_coeff = args.get('entropy_coeff', 0.01)

        # Compute entropy: H(π) = -Σ π × log π
        # probs is a list of [batch_size x n_nodes] probability distributions
        entropy = tf.constant(0.0)
        for p in probs:
            entropy += tf.reduce_mean(-tf.reduce_sum(p * tf.log(p + 1e-8), axis=1))
        entropy = entropy / len(probs)  # average over decode steps

        # Actor loss: advantage × log π - entropy bonus
        policy_loss = tf.reduce_mean(tf.multiply(advantage, tf.add_n(logprobs)), 0)
        actor_loss = policy_loss - entropy_coeff * entropy

        # Critic loss: MSE
        critic_loss = tf.losses.mean_squared_error(R, v)

        # Optimizers
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        # Gradients
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
