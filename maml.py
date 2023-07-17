import numpy as np

class MAML:
    def __init__(self, learner_class, params):
        self.learner_class = learner_class
        self.params = params
        self.model = None

    def sample_task_params(self):
        # Sample task-specific parameters
        task_params = {}
        for param in self.params.task_params:
            if param == 'lr_rate':
                task_params[param] = np.random.uniform(self.params.task_lr_rate_range[0], self.params.task_lr_rate_range[1])
            elif param == 'discount':
                task_params[param] = np.random.uniform(self.params.task_discount_range[0], self.params.task_discount_range[1])
            else:
                raise ValueError("Unsupported task parameter: %s" % param)
        return task_params

    def meta_update(self, tasks):
        if self.model is None:
            # Initialize the model with random initial parameters
            self.model = self.learner_class(self.params)
        
        # Perform meta-update for each task
        for task in tasks:
            learner = task[0](self.params)
            task_params = task[1]
            learner.set_params(task_params)

            # Generate trajectories using the current task
            trajs = []
            for i in range(self.params.num_traj_per_task):
                traj = get_traj(learner, self.params.env, self.params.episode_max_length)
                trajs.append(traj)

            all_ob = concatenate_all_ob(trajs, self.params)
            all_action = np.concatenate([traj["action"] for traj in trajs])
            all_adv = np.concatenate([traj["adv"] for traj in trajs])

            # Compute gradients and update the model parameters
            grads = learner.get_gradients(all_ob, all_action, all_adv)
            learner.update_parameters(grads)

        # Update the meta-learner model parameters using the accumulated gradients
        meta_grads = self.model.get_gradients(tasks)
        self.model.update_parameters(meta_grads)

    def evaluate(self, tasks):
        rewards = []
        for task in tasks:
            learner = task[0](self.params)
            task_params = task[1]
            learner.set_params(task_params)

            # Generate trajectories and compute rewards for evaluation
            trajs = []
            for i in range(self.params.num_traj_per_task):
                traj = get_traj(learner, self.params.env, self.params.episode_max_length)
                trajs.append(traj)

            rewards.append(compute_reward(trajs))

        return np.mean(rewards)

    def save_model(self, filename):
        if self.model is not None:
            # Save the model parameters to a file
            self.model.save(filename)
def get_traj(learner, env, episode_max_length):
    """
    Run learner-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []

    ob = env.observe()

    for _ in range(episode_max_length):
        act_prob = learner.get_action_prob(ob)
        a = np.random.choice(len(act_prob), p=act_prob)

        obs.append(ob)  # store the observation at the current decision-making step
        acts.append(a)

        ob, rew, done, _ = env.step(a)

        rews.append(rew)

        if done:
            break

    return {"reward": np.array(rews),
            "ob": np.array(obs),
            "action": np.array(acts)}


def concatenate_all_ob(trajs, params):
    timesteps_total = sum(len(traj["reward"]) for traj in trajs)

    all_ob = np.zeros(
        (timesteps_total, params.input_dim),
        dtype=np.float32)

    timesteps = 0
    for traj in trajs:
        traj_len = len(traj["reward"])
        all_ob[timesteps:timesteps + traj_len] = traj["ob"]
        timesteps += traj_len

    return all_ob


def compute_reward(trajs):
    rewards = [np.sum(traj["reward"]) for traj in trajs]
    return np.mean(rewards)
