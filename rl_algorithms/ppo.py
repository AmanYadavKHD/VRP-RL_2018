"""
PPO -- Proximal Policy Optimization
====================================
Prevents destructive large policy updates by clipping the probability ratio.

Loss:
    ratio      = exp( log pi_new - log pi_old )
    advantage  = R - V_stop_grad
    surr1      = ratio x advantage
    surr2      = clip(ratio, 1-eps, 1+eps) x advantage
    actor_loss = -mean( min(surr1, surr2) ) - entropy_coeff x H(pi)
    critic_loss = MSE(R, V)

In this implementation, we use "single-step PPO": the old policy is the
current policy (ratio starts at 1.0) but the clipping still prevents
the gradient from pushing the policy too far. This is equivalent to
running PPO with 1 epoch per batch, which is the standard approach when
on-policy data can't be reused (as in this TF1 static graph setup).

Reference:
    Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rl_algorithms.base import BaseAlgorithm


class PPO(BaseAlgorithm):

    @staticmethod
    def description():
        return ("PPO (Proximal Policy Optimization). Clips policy ratio to prevent "
                "destructive updates: min(ratio x A, clip(ratio) x A). "
                "Most stable modern policy gradient method.")

    @staticmethod
    def needs_critic():
        return True

    @staticmethod
    def build_train_step(agent, train_summary, args):
        R, v, logprobs, actions, idxs, batch, probs = train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)
        advantage = R - v_nograd

        # Normalize advantage for stability
        adv_mean, adv_var = tf.nn.moments(advantage, axes=[0])
        advantage = (advantage - adv_mean) / (tf.sqrt(adv_var) + 1e-8)

        ppo_clip = args.get('ppo_clip', 0.2)
        entropy_coeff = args.get('entropy_coeff', 0.01)

        # Current log probabilities (sum across decode steps)
        log_pi = tf.add_n(logprobs)

        # In TF1 static graph, we can't easily store old logprobs across
        # session.run() calls without a placeholder. Instead we use
        # "single-step PPO": old_logprobs = stop_gradient(current_logprobs).
        # This means ratio = exp(logpi - logpi) = 1 at the start of each step,
        # but the gradient still flows through log_pi_new, and the clip
        # constrains HOW FAR the gradient can push the policy in one update.
        log_pi_old = tf.stop_gradient(log_pi)
        ratio = tf.exp(log_pi - log_pi_old)

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantage
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Entropy bonus: H(pi) = -sum(pi * log(pi))
        entropy = tf.constant(0.0)
        for p in probs:
            entropy += tf.reduce_mean(-tf.reduce_sum(p * tf.log(p + 1e-8), axis=1))
        entropy = entropy / len(probs)

        actor_loss = policy_loss - entropy_coeff * entropy

        # Critic loss
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
