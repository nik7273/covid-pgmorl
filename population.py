import numpy as np
from copy import deepcopy

class SIR_Env:
    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass


class Population:
    def __init__(self):
        self.population = []
        self.models = []


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

    def prediction_model_candidates():
        """
        Receives:
        Outputs: candidate (policy, weight) pairs to use in task selection
        """
        # Prediction model used for calculation of expected objectives (see eq. (5) in paper)

        return candidates

        
    def prediction_guided_task_selection(num_tasks, ep, scalarization_template):
        """
        Receives: number of tasks to select `num_tasks` and pareto archive `ep` (see paper)
        Outputs: selected tasks `selected_tasks`
        """

        candidate_tasks = self.prediction_model_candidates()
        num_cands = len(candidate_tasks)

        # Task Selection (see Algorithm 3 in paper)
        task_mask = np.ones(num_cands, dtype=bool)
        virtual_ep_archive = deepcopy(ep)

        """ TODO """
        alpha = args.sparsity #????? what are they comparing to

        # best (task, weight) pairs and their advantage function weights
        selected_tasks, scalarization_batch = [], []
        for _ in range(num_tasks):
            hypervolumes = self.evaluate_hv(candidate_tasks, task_mask, virtual_ep_archive)
            sparsities = self.evaluate_sparsity(candidate_tasks, task_mask, virtual_ep_archive)

            # select maximizing (policy, weight) pair for Q(EP, T), where
            # Q(EP, T) = H(P) + alpha*S(P), where P is population,
            # H is hypervolume, and S is sparsity. See eq. 5 in paper.
            max_q, max_j = -np.inf, -1
            for j in range(num_cands):
                q = hypervolumes[j] - alpha * sparsities[j]
                if task_mask[j] and q > max_q:
                    max_q, max_j = q, j

            if max_j == -1:
                print("Too few candidates")
                break

            # add policies and weights to selected
            selected_tasks.append(candidate_tasks[max_j]['policy'])
            scalarization = deepcopy(scalarization_template)
            scalarization.update_weights(candidates[max_j]['weight'] / np.sum(candidates[best_id]['weight']))
            scalarization_batch.append(scalarization)
            
            task_mask[max_j] = False

            predicted_new_objs = [deepcopy(candidates[max_j]['prediction'])]
            new_objs_batch = np.array(virtual_ep_objs_batch + predicted_new_objs)
            virtual_ep_objs_batch = new_objs_batch[get_ep_indices(new_objs_batch)].tolist()

            predicted_offspring_objs.extend(predicted_new_objs)

        return selected_tasks, scalarization_batch, predicted_offspring_objs
    
