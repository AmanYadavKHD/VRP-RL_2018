def load_task_specific_components(task):
    '''
    This function load task-specific libraries
    - task: should be 'tsp' or 'vrp' (the task_name)
    '''
    if task == 'tsp':
        from TSP.tsp_utils import DataGenerator, Env ,reward_func
        from shared.attention import Attention

        AttentionActor = Attention
        AttentionCritic = Attention

    elif task == 'vrp':
        from VRP.vrp_utils import DataGenerator,Env,reward_func
        from VRP.vrp_attention import AttentionVRPActor,AttentionVRPCritic

        AttentionActor = AttentionVRPActor
        AttentionCritic = AttentionVRPCritic

    else:
        raise Exception(f'Task "{task}" is not implemented. Use "tsp" or "vrp".')

    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic
