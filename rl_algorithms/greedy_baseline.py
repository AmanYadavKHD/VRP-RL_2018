"""
Greedy Rollout Baseline
========================
Instead of a learned critic, uses the GREEDY policy's reward as baseline.

Loss:
    R_greedy   = reward from greedy decoding of the same input
    actor_loss = mean( (R_stochastic - R_greedy) × Σ log π )

No critic network is trained. The baseline comes from a "frozen" copy of
the policy that decodes greedily. The frozen policy is periodically updated
to match the current policy.

This was shown to OUTPERFORM learned critic baselines for routing problems
by Kool et al. (2019):
    "Attention, Learn to Solve Routing Problems!"
    https://arxiv.org/abs/1803.08475
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rl_algorithms.base import BaseAlgorithm


class GreedyBaseline(BaseAlgorithm):

    @staticmethod
    def description():
        return ("Greedy Rollout Baseline (Kool et al. 2019). No learned critic — "
                "uses greedy policy reward as baseline: (R_sample - R_greedy) × log π. "
                "Often outperforms learned critic for routing problems.")

    @staticmethod
    def needs_critic():
        """This algorithm doesn't need a learned critic."""
        return False

    @staticmethod
    def build_train_step(agent, train_summary, args):
        R_stoch, v, logprobs, actions, idxs, batch, probs = train_summary

        # Get greedy reward from the greedy decode built during model construction
        # agent.val_summary_greedy is already built: (R, v, logprobs, actions, idxs, batch, probs)
        R_greedy = agent.val_summary_greedy[0]  # R from greedy decoding

        R_stoch = tf.stop_gradient(R_stoch)
        R_greedy_sg = tf.stop_gradient(R_greedy)

        # Actor loss: (R_stochastic - R_greedy) × log π
        # When stochastic does worse than greedy (higher distance), this is positive,
        # so the gradient pushes AWAY from those actions.
        # When stochastic does better, this is negative, pushing TOWARD those actions.
        advantage = R_stoch - R_greedy_sg
        actor_loss = tf.reduce_mean(tf.multiply(advantage, tf.add_n(logprobs)), 0)

        # No critic loss
        critic_loss = tf.constant(0.0)

        # Optimizer (only actor)
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])

        actor_gra_and_var = actor_optim.compute_gradients(
            actor_loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'))

        clip_actor = [(tf.clip_by_norm(grad, args['max_grad_norm']), var)
                      for grad, var in actor_gra_and_var]

        actor_train_step = actor_optim.apply_gradients(clip_actor)

        # No critic training — use no-op
        critic_train_step = tf.no_op()

        # Return 0 for critic grad to keep format consistent
        critic_gra_and_var = [(tf.constant(0.0), tf.constant(0.0))]

        return [actor_train_step, critic_train_step,
                actor_loss, critic_loss,
                actor_gra_and_var, critic_gra_and_var,
                R_stoch, R_greedy, logprobs, probs, actions, idxs]
