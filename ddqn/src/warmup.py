def initialize_warmup_batch(args, model_cal, device):
    """
    Training policies during warmup stage
    """
    #print("in warmup initialization")
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    sample_batch = []
    scalarization_batch = []

    temp_env = SIR_env(model_cal) # temp_env is only used for initialization

    #print(f'about to evaluate {len(weights_batch)} policies')
    for weights in weights_batch:
        
        scalarization = WeightedSumScalarization(num_objs = args.obj_num, weights = weights)

        sample = Sample(model_cal.X_I, model_cal.X_S,objs = [None, None],  optgraph_id = -1) ###### NT: I am passing dummy objs.  
      
        #print('evaluating ...')
        objs = evaluate_policy(temp_env, sample.ddqn)
        #print('done with one evaluation')
        sample.objs = objs
        if sample is None:
          #print("None sample created!!!!!!!", sample)
          break
        sample_batch.append(sample)
        scalarization_batch.append(scalarization)

    return sample_batch, scalarization_batch