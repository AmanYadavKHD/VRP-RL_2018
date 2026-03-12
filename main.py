import argparse
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
from tqdm import tqdm 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

from configs import ParseParams, setup_logs

from shared.decode_step import RNNDecodeStep
from shared.task_utils import load_task_specific_components
from model.attention_agent import RLAgent
from rl_algorithms import get_algorithm, list_algorithms


def write_model_info(args, log_dir):
    """
    Write a human-readable model_info.txt to the log folder after training.
    Explains exactly what CSV format is needed to test this model.
    """
    task     = args.get('task', '?')
    task_name= args.get('task_name', '?')
    n_nodes  = args.get('n_nodes', '?')
    n_cust   = args.get('n_cust', '?')
    capacity = args.get('capacity', '?')
    dem_max  = args.get('demand_max', '?')
    dec_len  = args.get('decode_len', '?')
    hdim     = args.get('hidden_dim', '?')
    edim     = args.get('embedding_dim', '?')
    n_train  = args.get('n_train', '?')
    batch    = args.get('batch_size', '?')
    rl_model = args.get('rl_model', 'reinforce')
    model_dir= args.get('model_dir', log_dir + '/model')

    is_vrp = (task_name == 'vrp')
    input_dim = 3 if is_vrp else 2
    cols_per_node = input_dim
    total_data_cols = n_nodes * cols_per_node

    # Build example CSV header and one row
    if is_vrp:
        header_parts = ['problem_id']
        example_parts = ['1']
        for i in range(1, n_cust + 1):
            header_parts += [f'C{i}_x', f'C{i}_y', f'C{i}_demand']
            example_parts += [f'0.{i*7%100:02d}', f'0.{i*13%100:02d}', str((i*3 % dem_max) + 1)]
        header_parts += ['depot_x', 'depot_y', 'depot_demand']
        example_parts += ['0.50', '0.50', '0']
    else:
        header_parts = ['problem_id']
        example_parts = ['1']
        for i in range(1, n_nodes + 1):
            header_parts += [f'C{i}_x', f'C{i}_y']
            example_parts += [f'0.{i*7%100:02d}', f'0.{i*13%100:02d}']

    header_line = ','.join(header_parts)
    example_line = ','.join(str(v) for v in example_parts)

    lines = []
    lines.append('=' * 68)
    lines.append(f'  MODEL INFO -- {task}  ({log_dir})')
    lines.append('=' * 68)
    lines.append('')
    lines.append('This file tells you everything about this trained model:')
    lines.append('what it solves, what constraints it has, and how to test it')
    lines.append('with your own custom data.')
    lines.append('')

    lines.append('-' * 68)
    lines.append('  WHAT THIS MODEL SOLVES')
    lines.append('-' * 68)
    if is_vrp:
        lines.append(f'  Problem type : VRP (Vehicle Routing Problem)')
        lines.append(f'  Customers    : {n_cust} delivery stops per problem')
        lines.append(f'  Nodes        : {n_nodes} total (customers + 1 depot)')
        lines.append(f'  Vehicle cap  : {capacity}  (max total demand per vehicle trip)')
        lines.append(f'  Max demand   : {dem_max}  (max demand any single customer can have)')
        lines.append(f'  Decode steps : {dec_len}  (max route length)')
    else:
        lines.append(f'  Problem type : TSP (Travelling Salesman Problem)')
        lines.append(f'  Cities       : {n_nodes} per problem')
        lines.append(f'  Decode steps : {dec_len}')
    lines.append(f'  RL Algorithm : {rl_model.upper()}')
    lines.append('')

    lines.append('-' * 68)
    lines.append('  COORDINATE SPACE')
    lines.append('-' * 68)
    lines.append('  x and y values MUST be between 0.0 and 1.0')
    lines.append('  Think of it as a 1km x 1km square area.')
    lines.append('  Real coordinates: normalize them first!')
    lines.append('')
    lines.append('  Example: city at lat=28.7, lon=77.1 in a region 28.5-29.0 lat, 76.8-77.4 lon:')
    lines.append('    x = (77.1 - 76.8) / (77.4 - 76.8) = 0.50')
    lines.append('    y = (28.7 - 28.5) / (29.0 - 28.5) = 0.40')
    lines.append('')

    if is_vrp:
        lines.append('-' * 68)
        lines.append('  VRP CONSTRAINTS')
        lines.append('-' * 68)
        lines.append(f'  Vehicle capacity : {capacity}')
        lines.append(f'    Each vehicle can carry at most {capacity} units of goods.')
        lines.append(f'    When it runs out, it returns to depot and refills.')
        lines.append(f'  Customer demand  : 1 to {dem_max}')
        lines.append(f'    Each customer needs between 1 and {dem_max} units.')
        lines.append(f'  Depot demand     : MUST be 0  (depot is always the last node)')
        lines.append(f'  Number of vehicles: unlimited (model uses as many as needed)')
        lines.append('')

    lines.append('-' * 68)
    lines.append('  HOW TO BUILD YOUR OWN TEST CSV')
    lines.append('-' * 68)
    lines.append(f'  File location : custom_testing/my_test.csv')
    lines.append(f'  Format        : CSV with header row, one problem per data row')
    lines.append(f'  Columns       : {total_data_cols + 1} total  (1 problem_id + {total_data_cols} data columns)')
    lines.append(f'  Data columns  : {n_nodes} nodes x {cols_per_node} values = {total_data_cols} columns')
    lines.append('')
    if is_vrp:
        lines.append('  Column order per node (repeat for each node):')
        lines.append('    node_x   - x coordinate [0.0 to 1.0]')
        lines.append('    node_y   - y coordinate [0.0 to 1.0]')
        lines.append(f'    node_demand - integer [1 to {dem_max}] for customers, 0 for depot')
        lines.append('')
        lines.append(f'  Node order: C1, C2, ..., C{n_cust}, DEPOT')
        lines.append(f'  LAST node is always the DEPOT  (demand must be 0)')
    else:
        lines.append('  Column order per node:')
        lines.append('    node_x - x coordinate [0.0 to 1.0]')
        lines.append('    node_y - y coordinate [0.0 to 1.0]')
    lines.append('')
    lines.append('  EXAMPLE CSV (copy-paste and modify):')
    lines.append('')
    lines.append('  ' + header_line)
    lines.append('  ' + example_line)
    lines.append('  ' + example_line.replace(',1,', ',2,').replace('problem_id', '').lstrip(','))
    lines.append('')
    lines.append(f'  See also: custom_testing/example_{task}.csv  (ready-made template)')
    lines.append('')

    lines.append('-' * 68)
    lines.append('  TERMINAL COMMANDS')
    lines.append('-' * 68)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    ckpt_str = latest_ckpt if latest_ckpt else f'{model_dir}/model.ckpt-0'
    lines.append('')
    lines.append('  # 1. Run routes on the default test data (auto-selected):')
    lines.append(f'  python view_routes.py --is_train=False --task={task}')
    lines.append('')
    lines.append('  # 2. Run routes on the default test data with THIS specific model:')
    lines.append(f'  python view_routes.py --is_train=False --task={task} ^')
    lines.append(f'      --load_path "{ckpt_str}"')
    lines.append('')
    lines.append('  # 3. Run with your own CSV:')
    lines.append(f'  python view_routes.py --is_train=False --task={task} ^')
    lines.append(f'      --load_path "{ckpt_str}" ^')
    lines.append('      --csv_path custom_testing/my_test.csv')
    lines.append('')
    lines.append('  # 4. Control number of problems shown in routes.png:')
    lines.append(f'  python view_routes.py --is_train=False --task={task} ^')
    lines.append(f'      --load_path "{ckpt_str}" ^')
    lines.append('      --n_show 10')
    lines.append('')
    lines.append('  # 5. Analyze training results (charts + summary):')
    lines.append(f'  python analyze_results.py --log_dir {log_dir}')
    lines.append('')

    lines.append('-' * 68)
    lines.append('  MODEL ARCHITECTURE')
    lines.append('-' * 68)
    lines.append(f'  Hidden dim   : {hdim}')
    lines.append(f'  Embedding dim: {edim}')
    lines.append(f'  Train steps  : {n_train}')
    lines.append(f'  Batch size   : {batch}')
    lines.append('')
    lines.append('  NOTE: This model can ONLY be used for the same task it was')
    lines.append(f'  trained on: {task} ({n_cust} customers, capacity {capacity}).')
    lines.append('  A vrp10 model cannot route vrp20 problems.')
    lines.append('')
    lines.append('=' * 68)

    out_path = os.path.join(log_dir, 'model_info.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('[OK] Model info written to: {}'.format(out_path))

def run_automated_analysis(log_dir, args):
    """Run analyze_results.py and view_routes.py logic automatically."""
    try:
        # 1. Statistical Analysis & Phase Plots (training_plots.png)
        import analyze_results
        data = analyze_results.parse_results(log_dir)
        if data:
            analyze_results.plot_results(data, log_dir)
            summary_path = os.path.join(log_dir, "analysis_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                import sys
                original_stdout = sys.stdout
                sys.stdout = f
                try:
                    analyze_results.print_summary(data, log_dir)
                finally:
                    sys.stdout = original_stdout
        # 2. Route Visualization (routes.png)
        # We run this in a separate process to avoid any resource/session conflicts with the training process
        import subprocess
        model_dir = os.path.join(log_dir, "model")
        ckpt = tf.train.latest_checkpoint(model_dir)
        if ckpt:
            print(f"Triggering route visualization for: {ckpt}")
            cmd = [
                sys.executable, 
                "view_routes.py", 
                "--task", str(args['task']),
                "--load_path", str(ckpt),
                "--n_show", "4",
                "--is_train", "False"
            ]
            subprocess.run(cmd, check=False)
            print(f"[✓] Automation cycle finished.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] Could not run automated analysis/routes: {e}")

# Removed: load_task_specific_components moved to shared.task_utils

def main(args, prt):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load task specific classes
    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = \
        load_task_specific_components(args['task_name'])

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)

    # load RL algorithm
    rl_model_name = args.get('rl_model', 'reinforce')
    rl_algorithm = get_algorithm(rl_model_name)
    prt.print_out('RL Algorithm: {} -- {}'.format(rl_model_name, rl_algorithm.description()))

    # create an RL agent
    agent = RLAgent(args,
                    prt,
                    env,
                    dataGen,
                    reward_func,
                    AttentionActor,
                    AttentionCritic,
                    is_train=args['is_train'],
                    rl_algorithm=rl_algorithm)
    agent.Initialize(sess)

    # train or evaluate
    start_time = time.time()
    history = []  # List of (step, avg_reward, avg_value)
    
    if args['is_train']:
        prt.print_out('Training started ...')
        train_time_beg = time.time()
        for step in range(args['n_train']):
            summary = agent.run_train_step()
            _, _ , actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val,\
                R_val, v_val, logprobs_val,probs_val, actions_val, idxs_val= summary

            if step%args['save_interval'] == 0:
                agent.saver.save(sess,args['model_dir']+'/model.ckpt', global_step=step)

            if step%args['log_interval'] == 0:
                train_time_end = time.time()-train_time_beg
                avg_r = np.mean(R_val)
                avg_v = np.mean(v_val)
                history.append((step, avg_r, avg_v))
                
                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'\
                      .format(step,time.strftime("%H:%M:%S", time.gmtime(\
                        train_time_end)),avg_r,avg_v))
                prt.print_out('    actor loss: {} -- critic loss: {}'\
                      .format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                train_time_beg = time.time()
                agent.inference(args['infer_type'])

        # Save final checkpoint before automation
        final_ckpt = os.path.join(args['model_dir'], 'model.ckpt')
        agent.saver.save(sess, final_ckpt, global_step=step)
        print(f"[✓] Final checkpoint saved: {final_ckpt}")

        # Save final analysis & routes
        run_automated_analysis(args['log_dir'], args)

    else: # inference
        prt.print_out('Evaluation started ...')
        agent.inference(args['infer_type'])


    prt.print_out('Total time is {}'.format(\
        time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))

    # Write model documentation to log folder
    if args['is_train']:
        write_model_info(args, args['log_dir'])

if __name__ == "__main__":
    args = ParseParams()
    
    # Only setup logs/folders if we are actually training
    prt = None
    if args['is_train']:
        args, prt = setup_logs(args)
    else:
        # For inference, just print to console, don't create new timestamped folders
        from shared.misc_utils import printOut
        prt = printOut(None, True)

    # Random
    random_seed = args['random_seed']
    tf.reset_default_graph()
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    main(args, prt)
