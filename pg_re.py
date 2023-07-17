import time
import numpy as np
import theano
import cPickle
import matplotlib.pyplot as plt

import environment
import job_distribution
import pg_network
import slow_down_cdf

from maml import MAML


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in xrange(episode_max_length):
        act_prob = agent.get_one_act_prob(ob)
        csprob_n = np.cumsum(act_prob)
        a = (csprob_n > np.random.rand()).argmax()

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: break

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'entropy': entropy,
            'info': info
            }


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in xrange(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in xrange(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in xrange(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def main():
    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 50  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    envs = []
    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    for ex in xrange(pa.num_ex):
        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre='image', end='all_done')
        env.seq_no = ex
        envs.append(env)

    maml = MAML(pg_network.PGLearner, pa)

    # Start meta-training
    for iteration in xrange(1, pa.num_epochs+1):
        print("-----------------")
        print("Meta-Training Iteration: %i" % iteration)

        # Generate tasks for meta-update
        tasks = []
        for _ in range(pa.batch_size):
            task_params = maml.sample_task_params()
            task = (pg_network.PGLearner, task_params)
            tasks.append(task)

        # Perform meta-update
        maml.meta_update(tasks)

        # Evaluate on unseen tasks
        unseen_tasks = []
        for _ in range(pa.batch_size):
            task_params = maml.sample_task_params()
            task = (pg_network.PGLearner, task_params)
            unseen_tasks.append(task)

        avg_reward = maml.evaluate(unseen_tasks)
        print("Average reward on unseen tasks: %.2f" % avg_reward)

        print("-----------------")

    # Save the final model
    maml.save_model(pa.output_filename)

    # Evaluate on unseen examples
    pa.unseen = True
    slow_down_cdf.launch(pa, pa.output_filename, render=False, plot=True, repre='image', end='all_done')
    pa.unseen = False

    print("Training completed.")


if __name__ == '__main__':
    main()
