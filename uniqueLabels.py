import glob


chars = set()
# Scan all text files in the dataset folder
for filepath in glob.glob("data/ICDAR/targets.txt"):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            chars.update(line)  # add all characters from the line


print(chars)


def merge_best_states(global_model, best_dicts, alphas=None):
    if alphas is None:
        alphas = [1.0 / len(best_dicts)] * len(best_dicts)  # equal weighting

    # start from global_model state_dict
    merged_state = copy.deepcopy(global_model.state_dict())

    # merge each parameter
    for key in merged_state.keys():
        merged_state[key] = sum(alpha * bd[key] for alpha, bd in zip(alphas, best_dicts))

    # load merged weights into global_model
    global_model.load_state_dict(merged_state)
    return global_model
 
