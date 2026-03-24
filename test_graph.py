"""
Quick dry-run test: builds the TF graph and runs ONE training step.
If this passes, the full training will work.
Usage: python test_graph.py --task=vrp10
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import traceback

from configs import ParseParams
from shared.task_utils import load_task_specific_components
from model.attention_agent import RLAgent
from rl_algorithms import get_algorithm

def test():
    # Parse args
    args = ParseParams()
    args['n_train'] = 1
    args['batch_size'] = 4   # small batch for quick test
    args['test_size'] = 4
    args['is_train'] = True
    
    # Suppress log folder creation
    args['log_dir'] = 'logs/test_dry_run'
    args['model_dir'] = 'logs/test_dry_run/model'
    os.makedirs(args['model_dir'], exist_ok=True)
    
    from shared.misc_utils import printOut
    prt = printOut(None, True)

    print("=" * 60)
    print(f"  DRY RUN TEST: {args['task']}")
    print(f"  batch_size={args['batch_size']}, n_nodes={args['n_nodes']}")
    print("=" * 60)

    tf.reset_default_graph()

    # Load components
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    print("[1/6] Creating DataGenerator...")
    dataGen = DataGenerator(args)
    dataGen.reset()

    print("[2/6] Creating Env...")
    env = Env(args)

    print("[3/6] Loading RL algorithm...")
    rl_algorithm = get_algorithm(args.get('rl_model', 'reinforce'))

    print("[4/6] Building RLAgent (greedy + beam_search + stochastic graphs)...")
    try:
        agent = RLAgent(args, prt, env, dataGen, reward_func,
                        AttentionActor, AttentionCritic, 
                        is_train=True, rl_algorithm=rl_algorithm)
        print("  ✓ All 3 graphs built successfully!")
    except Exception as e:
        print(f"  ✗ GRAPH BUILD FAILED: {e}")
        traceback.print_exc()
        return False

    print("[5/6] Initializing session...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    agent.Initialize(sess)
    print("  ✓ Session initialized, weights loaded")

    print("[6/6] Running ONE training step...")
    try:
        data = dataGen.get_train_next()
        print(f"  Train data shapes: input_pnt={data[0].shape}", end="")
        if len(data) > 2:
            print(f", demand={data[1].shape}, time_matrix={data[2].shape}")
        else:
            print(f", time_matrix={data[1].shape}")
        
        summary = agent.run_train_step()
        _, _, actor_loss, critic_loss, _, _, R, v, _, _, _, _ = summary
        print(f"  ✓ Training step completed!")
        print(f"    Reward (avg):     {np.mean(R):.2f}")
        print(f"    Value (avg):      {np.mean(v):.2f}")
        print(f"    Actor loss:       {np.mean(actor_loss):.6f}")
        print(f"    Critic loss:      {np.mean(critic_loss):.6f}")
    except Exception as e:
        print(f"  ✗ TRAINING STEP FAILED: {e}")
        traceback.print_exc()
        sess.close()
        return False

    # Test inference too
    print("\n[BONUS] Testing inference (greedy)...")
    try:
        agent.inference('batch')
        print("  ✓ Inference completed!")
    except Exception as e:
        print(f"  ✗ INFERENCE FAILED: {e}")
        traceback.print_exc()
        sess.close()
        return False

    sess.close()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("  You can now run full training:")
    print(f"  python main.py --task={args['task']} --is_train=True --n_train=500 --batch_size=32 --log_interval=50")
    print("=" * 60)
    return True

if __name__ == '__main__':
    test()
