### Main MORL program

def morl(args): # num_parallel_tasks, warmup_iters, evo_iters, num_generations
    print("In run")
    """
    Runs the entire MORL algorithm. (See Alg. 1 in Xu et al.)
    """
    # Initalization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cuda")

    '''environment (Not done)'''
    

    '''gives : model_cal, the calibrated model'''

    scalarization_template = WeightedSumScalarization(num_objs = args.obj_num, weights = np.ones(args.obj_num) / args.obj_num)
    total_num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    
    external_pareto = EP()
    population = Population()
    opt_graph = OptGraph()

    model_cal = cal_model
    print('initializing warmup batch')
    selected_tasks, scalarization_batch = initialize_warmup_batch(args, model_cal, device)
    print('done initializing warmup batch')
    rl_num_updates = args.warmup_iter
    eval_num = args.eval_num

    for sample, scalarization in zip(selected_tasks, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(deepcopy(scalarization.weights), deepcopy(sample.objs), -1)
    print('done inserting to opt_graph')

    #Evolutionary Stage
    episode = 0
    iteration = 0
    print("Done initializing")
    total_batch=[]
    iteration = 0
    i=0
    last_scalarization_batch = []
    last_offspring_batch = []
    print(iteration, args.num_generations, total_num_updates,args.warmup_iter, args.update_iter,i)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #for i in range(args.num_generations):
    while iteration < total_num_updates:
        print(f"In iteration {iteration}")
        if episode == 0:
            print_info('\n------------------------------- Warm-up Stage -------------------------------')    
        else:
            print_info('\n-------------------- Evolutionary Stage: Generation {:3} --------------------'.format(episode))
        
        episode += 1

        offspring_batch = np.array([])

        # --------------------> RL Optimization <-------------------- #
        # compose task for each elite
        task_batch = []
        for selected, scalarization in zip(selected_tasks, scalarization_batch):
            task_batch.append(Task(selected, scalarization)) # each task is a (policy, weight)

        all_offspring_batch = [[] for _ in range(len(task_batch))]

        env = SIR_env(model_cal) 
        offspring_population = mopg(env, task_batch, rl_num_updates, eval_num)
        for task_id, sample in enumerate(offspring_population):
            all_offspring_batch[task_id].append(Sample.copy_from(sample))

        # put all intermediate policies into all_sample_batch for EP update
        all_sample_batch = []
        offspring_batch = []
        last_offspring_batch = [None] * len(task_batch) 
        for task_id in range(len(task_batch)):
            #print(f'Running intermediate policies: Task id: {task_id}/{len(task_batch)}')
            offsprings = all_offspring_batch[task_id]
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()

            for k, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                sample.optgraph_id = prev_node_id
                offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]
        
        total_batch.append(last_offspring_batch)
        # ----------------------> Update EP <------------------------ #
        #print("Updating EP :O")
        external_pareto.update(all_sample_batch)
        
        population.update(offspring_batch)
        # -------------------> Task Selection for Next Stage/Evaluation <--------------------- #
        last_scalarization_batch = scalarization_batch
        selected_samples, scalarization_batch, predicted_offspring_objs = \
            population.prediction_guided_task_selection(args, external_pareto, opt_graph, scalarization_template)
        selected_tasks = deepcopy(selected_samples)
        obj_cost = np.zeros((len(selected_samples), 2))
        obj_wts = np.zeros((len(selected_samples), 2))
        print_info('Selected Tasks:')
        for j in range(len(selected_samples)):
            #print_info('objs = {}, weight = {}'.format(selected_samples[j].objs, scalarization_batch[j].weights))
            print_info(F"objs = [{selected_samples[j].objs[0] }, {selected_samples[j].objs[1]}]" +  # *  model_cal.M
                       F" weight = {scalarization_batch[j].weights}")
            obj_cost[j,:] = np.array([np.array([selected_samples[j].objs[0],selected_samples[j].objs[1]])])
            obj_wts[j,:] = np.array([scalarization_batch[j].weights.numpy()])

        print(obj_cost)
        print(obj_wts)

        np.save(os.path.join(args.save_dir, "obj_cost_%i" % iteration), obj_cost)
        np.save(os.path.join(args.save_dir, "obj_wts_%i" % iteration), obj_wts)

        for jj, offspring in enumerate(last_offspring_batch):
            torch.save(offspring.ddqn, os.path.join(args.save_dir, "ddqn%i.pt" % jj))
            torch.save(offspring.ddqn.NN, os.path.join(args.save_dir, "ddqn%i_NN.pt" % jj))
            torch.save(offspring.ddqn.target_NN, os.path.join(args.save_dir, "ddqn%i_target_NN.pt" % jj))
        
        iteration = min(iteration + rl_num_updates, total_num_updates)

        rl_num_updates = args.update_iter
        print(iteration, args.num_generations, total_num_updates,args.warmup_iter, args.update_iter,i)

    print('begin evaluation and plots')

    policy_radial_heatmap(last_offspring_batch,last_scalarization_batch,args)

    print("MORL DONE!")
    print('Post-processing')
