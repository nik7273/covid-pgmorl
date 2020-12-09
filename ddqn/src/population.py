"""
Most of the following is taken from https://github.com/mit-gfx/PGMORL/, with minor changes
"""
def collect_nearest_data(opt_graph, optgraph_id, threshold = 0.1):
    objs_data, weights_data, delta_objs_data = [], [], []
    for i in range(len(opt_graph.objs)):
        diff = np.abs(opt_graph.objs[optgraph_id] - opt_graph.objs[i])
        
        if np.all(diff < np.abs( opt_graph.objs[optgraph_id]) * threshold):
            for next_index in opt_graph.succ[i]:
                objs_data.append(opt_graph.objs[i])
                weights_data.append(opt_graph.weights[next_index] / np.sum(opt_graph.weights[next_index]))
                delta_objs_data.append(opt_graph.delta_objs[next_index])
    return objs_data, weights_data, delta_objs_data

def predict_hyperbolic(args, opt_graph, optgraph_id, test_weights):
    """
    Taken from https://github.com/mit-gfx/PGMORL/
    Receives: optimization graph, policy identifier, weights for test run
    Outputs: results, which is a dict with policy and predictions for that policy
    Effects: Trains hyperbolic prediction function for a policy of optgraph_id.
    """
    test_weights = np.array(test_weights)

    # normalize the test_weights to be sum = 1
    for test_weight in test_weights:
        test_weight /= np.sum(test_weight)
    
    threshold = deepcopy(np.float32(args.hbolic_init_thresh))
    factor = deepcopy(np.float32(args.hbolic_thresh_factor))
    max_threshold = deepcopy(np.float32(args.hbolic_max_thresh))    

    
    sigma = 0.03
    loop_iterations = 0 # dummy variable, no use.
    # gradually enlarging the searching range so that get enough data point to fit the model

    #print(opt_graph.objs[optgraph_id])

    while True:
        objs_data, weights_data, delta_objs_data = collect_nearest_data(opt_graph, optgraph_id, threshold)
        cnt_data = 0

        for i in range(len(weights_data)):
            flag = True
            for j in range(i):
                if np.linalg.norm(weights_data[i] - weights_data[j]) < 1e-5:
                    flag = False
                    break
            if flag:
                cnt_data += 1
                if cnt_data > 3:
                    break
        if cnt_data > 3:
            break
        else:
            sigma *= 2.0
            if threshold <= max_threshold:
                threshold *= factor
            

    def f(x, A, a, b, c):
        return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

    def fun(params, x, y):
        # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
        return (params[0] * (np.exp(params[1] * (x - params[2])) - 1.) / (np.exp(params[1] * (x - params[2])) + 1) + params[3] - y) * w

    def jac(params, x, y):
        A, a, b, c = params[0], params[1], params[2], params[3]

        J = np.zeros([len(params), len(x)])

        # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
        J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w

        # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[1] = (A * (x - b) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
        J[2] = (A * (-a) * (2. * np.exp(a * (x - b))) / ((np.exp(a * (x - b)) + 1) ** 2)) * w

        # df_dc = 1
        J[3] = w

        return np.transpose(J)

    M = args.obj_num
    delta_predictions = []
    for dim in range(M):
        train_x = []
        train_y = []
        w = []
        for i in range(len(objs_data)):
            train_x.append(weights_data[i][dim])
            train_y.append(delta_objs_data[i][dim])
            diff = np.abs(objs_data[i] - opt_graph.objs[optgraph_id])
            dist = np.linalg.norm(diff / np.abs(opt_graph.objs[optgraph_id]))
            coef = np.exp(-((dist  / sigma) ** 2) / 2.0)
            w.append(coef)
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        w = np.array(w)

        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        params0 = np.ones(4)
        
        f_scale = 20.

        # fit the prediction function by minimizing soft_l1 loss.
        res_robust = least_squares(
            fun,
            params0,
            loss='soft_l1',
            f_scale = f_scale,
            args = (train_x, train_y),
            jac = jac,
            bounds = ([0, 0.1, -5., -500.], [A_upperbound, 20., 5., 500.])
        )
        
        delta_predictions.append(f(test_weights.T[dim], *res_robust.x))

    predictions = []
    delta_predictions = np.transpose(np.array(delta_predictions))
    original_objs = opt_graph.objs[optgraph_id]
    for i in range(len(test_weights)):
        predictions.append(original_objs + delta_predictions[i])

    results = {'sample_index': optgraph_id, 'predictions': predictions}

    return results
    

class Population:
    def __init__(self):
        self.population = []
        #self.models = []

        #Added code for pbuffer
        self.pbuffer_num = args.pbuffer_num #index of the buffer
        self.pbuffer_size = args.pbuffer_size
        self.dtheta = (np.pi / 2.0) / self.pbuffer_num
        self.z_min = np.zeros(args.obj_num) # reference point
        self.pbuffers = None
        self.pbuffer_dist = None

    """
    insert the sample to the performance buffers (storing the index).
    """

    def insert_pbuffer(self, index, objs):
        f = objs - self.z_min
        if np.min(f) < 1e-7:
            return False
        
        dist = np.linalg.norm(f)
        theta = np.arccos(np.clip(f[1] / dist, -1.0, 1.0))
        buffer_id = int(theta // self.dtheta)
        if buffer_id < 0 or buffer_id >= self.pbuffer_num:
            return False

        inserted = False
        # insert sample into the corresponding pbuffer if its distance to origin is top pbuffer_size
        # store the samples in each pbuffer in order of distance
        for i in range(len(self.pbuffers[buffer_id])):
            if self.pbuffer_dist[buffer_id][i] < dist:
                self.pbuffers[buffer_id].insert(i, index)
                self.pbuffer_dist[buffer_id].insert(i, dist)
                inserted = True
                break
        if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
            self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
            self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
        elif (not inserted) and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
            self.pbuffers[buffer_id].append(index)
            self.pbuffer_dist[buffer_id].append(dist)
            inserted = True

        return inserted

    '''
    update the population by a new offspring sample_batch.
    '''  
    def update(self, given_pop):
        ### population = Union(population, offspring) ###
        all_pop = self.population + given_pop
        print(f'Length of original population {len(self.population)}')
        self.population = []
        self.pbuffers = [[] for _ in range(self.pbuffer_num)]       # store the sample indices in each pbuffer
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)]   # store the sample distance in each pbuffer

        ### select the population by performance buffer ###       
        for i, sample in enumerate(all_pop):
            self.insert_pbuffer(i, sample.objs)

        print("should print that its updating")        
        for pbuffer in self.pbuffers:
            for index in pbuffer:
                print('appending to population in update function')
                self.population.append(all_pop[index])

    def compute_hypervolume(self, objs_batch):
        """
        Taken from https://github.com/mit-gfx/PGMORL/
        Receives:
        Outputs: hypervolume (see eq. 1 in paper)
        """
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        ref_x, ref_y = 0.0, 0.0
        x, hv = ref_x, 0.0
        for objs in ep_objs_batch:
            hv += (max(ref_x, objs[0]) - x) * (max(ref_y, objs[1]) - ref_y)
            x = max(ref_x, objs[0])
        return hv

    def compute_sparsity(self, objs_batch):
        """
        Taken from https://github.com/mit-gfx/PGMORL/
        Receives:
        Outputs: sparsity (see eq. 2 in paper)
        """
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        if len(ep_objs_batch) < 2:
            return 0.0
        sparsity = 0.0
        for i in range(1, len(ep_objs_batch)):
            sparsity += np.sum(np.square(ep_objs_batch[i] - ep_objs_batch[i - 1]))
        sparsity /= (len(ep_objs_batch) - 1)
        return sparsity
    
    def evaluate_hv(self, candidates, mask, virtual_ep_objs_batch):
        """
        Receives:
        Outputs: hypervolume set for each candidate (task, weight) pair after
        virtually inserting predicted offspring.
        """
        hv = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
                hv[i] = self.compute_hypervolume(new_objs_batch)
        return hv

    def evaluate_sparsity(self, candidates, mask, virtual_ep_objs_batch):
        """
        Receives:
        Outputs: sparsity set for each candidate (task, weight) pair after
        virtually inserting predicted offspring.
        """
        sparsity = [0.0 for _ in range(len(candidates))]
        for i in range(len(candidates)):
            if mask[i]:
                new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
                sparsity[i] = self.compute_sparsity(new_objs_batch)     
        return sparsity        

    def prediction_model_candidates(self, args, opt_graph):
        """
        Receives: 
        Outputs: candidate (policy, weight) pairs to use in task selection
        """
        # Prediction model used for calculation of expected objectives (see eq. (5) in paper)
        num_weights = args.num_weights_candidates
        
        #List of tasks to 
        policy_sample_batch = self.population
        print("Length of population:")
        print(len(policy_sample_batch))

        candidates = []
        for policy in policy_sample_batch:
            weight_center = opt_graph.weights[policy.optgraph_id]
            angle_center = np.arctan2(weight_center[1], weight_center[0])
            angle_bound = [angle_center - np.pi / 4., angle_center + np.pi / 4.]
            test_weights = []
            for i in range(num_weights):
                angle = angle_bound[0] + ((angle_bound[1] - angle_bound[0]) / (num_weights - 1)) * i
                weight = np.array([np.cos(angle), np.sin(angle)])
                if weight[0] >= -1e-7 and weight[1] >= -1e-7:
                    duplicated = False
                    for succ in opt_graph.succ[policy.optgraph_id]: # discard duplicate tasks

                        w = deepcopy(opt_graph.weights[succ])
                        w = w / np.linalg.norm(w)

                        if np.linalg.norm(w - weight) < 1e-3:
                            print('marking duplicated')
                            duplicated = True
                            break
                    
                    if not duplicated:
                        print('weigths are not duplicated. appending weight', weight, F'for {i} in {num_weights}.')
                        test_weights.append(weight)

            if len(test_weights) > 0:
                results = predict_hyperbolic(args, opt_graph, policy.optgraph_id, test_weights)
                for i in range(len(test_weights)):
                    print('Adding to candidates')
                    candidates.append({'policy': policy, 'weight': test_weights[i], \
                        'prediction': results['predictions'][i]})
            
        return candidates
        
    def prediction_guided_task_selection(self, args, ep, opt_graph, scalarization_template):
        """
        Receives: number of tasks to select `num_tasks` and pareto archive `ep` (see paper)
        Outputs: selected tasks `selected_tasks`
        """
        print('in prediction_guided_task_selection')
        candidates = self.prediction_model_candidates(args, opt_graph) #need to add args
        print(f'There are {len(candidates)} candidates')
        print('after obtaining candidates, in prediction_guided_task_selection')

        #initialize virtual ep
        virtual_ep_objs_batch = []
        for i in range(len(ep.sample_batch)):
            virtual_ep_objs_batch.append(deepcopy(ep.sample_batch[i].objs))

        num_cands = len(candidates)
        num_tasks = args.num_tasks

        # Task Selection (see Algorithm 3 in paper)
        task_mask = np.ones(num_cands, dtype=bool)

        predicted_offspring_objs = []

        alpha = args.sparsity 

        # best (task, weight) pairs and their advantage function weights
        selected_samples, scalarization_batch = [], []
        for i in range(num_tasks):
            print(f'running in prediction selected task {i+1}/{num_tasks}')
            '''MIT implementation https://github.com/mit-gfx/PGMORL/blob/master/morl/population_2d.py'''
            hypervolumes = self.evaluate_hv(candidates, task_mask, virtual_ep_objs_batch)
            sparsities = self.evaluate_sparsity(candidates, task_mask, virtual_ep_objs_batch)
            # select maximizing (policy, weight) pair for Q(EP, T), where
            # Q(EP, T) = H(P) + alpha*S(P), where P is population,
            # H is hypervolume, and S is sparsity. See eq. 5 in paper.
            max_q, max_j = -np.inf, -1
            for j in range(num_cands):
                #print('before sus multiply')
                q = hypervolumes[j] - alpha * sparsities[j] ### POTENTIAL ERROR HERE - WHY SPARSITY METRIC IS NEGATED??? ### They are right, the sparsity is minimized if dots in Pareto are evenly distributed. (That's why they have a square)
                ###We can reduce alpha to allow larger sparsities.
                if task_mask[j] and q > max_q:
                    max_q, max_j = q, j

            if max_j == -1:
                print("Too few candidates")
                break

            # add policies and weights to selected
            selected_samples.append(candidates[max_j]['policy'])
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(candidates[max_j]['weight'] / np.sum(candidates[max_j]['weight']))
            scalarization_batch.append(scalarization)
            
            task_mask[max_j] = False

            predicted_new_objs = [deepcopy(candidates[max_j]['prediction'])]
            new_objs_batch = np.array(virtual_ep_objs_batch + predicted_new_objs)
            virtual_ep_objs_batch = new_objs_batch[get_ep_indices(new_objs_batch)].tolist()

            predicted_offspring_objs.extend(predicted_new_objs)

        return selected_samples, scalarization_batch, predicted_offspring_objs#predicted_new_objs

