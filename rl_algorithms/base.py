"""
Base class for RL training algorithms.

All algorithms must implement:
  - build_train_step(agent, train_summary, args) → train_op list
  - description() → human-readable string
  - needs_critic() → whether this algorithm needs a learned critic network
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class BaseAlgorithm:
    """Abstract base class for RL training algorithms."""

    @staticmethod
    def description():
        raise NotImplementedError

    @staticmethod
    def needs_critic():
        """If True, the critic network is built during model construction."""
        return True

    @staticmethod
    def build_train_step(agent, train_summary, args):
        """
        Build the training operation.

        Args:
            agent: RLAgent instance (has decodeStep, env, etc.)
            train_summary: tuple (R, v, logprobs, actions, idxs, batch, probs)
            args: hyperparameter dict

        Returns:
            list of ops to run per training step:
            [actor_train_op, critic_train_op, actor_loss, critic_loss,
             actor_grad_var, critic_grad_var, R, v, logprobs, probs, actions, idxs]
        """
        raise NotImplementedError
