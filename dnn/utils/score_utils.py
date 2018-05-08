import numpy as np

def decision_cost(sim_array, ths, labels, cost_miss=10, cost_fa=1, p_target=0.01):
    decision = sim_array > ths
    incorrect_decisions = (decision != labels)
    non_targets = (labels == 0)
    targets = (labels == 1)
    fa_rate = np.sum(incorrect_decisions[non_targets]) / np.sum(decision)
    miss_rate = np.sum(incorrect_decisions[targets]) / np.sum(~decision)
    cost = cost_miss * miss_rate * p_target + cost_fa * fa_rate * (1-p_target)
    return cost

