import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics

def compute_matches(file_train, file_test, alphabet, n, r):
    # train using the given training file, and compute the scores for the given test file
    cmd = f'java -jar negsel2.jar -alphabet {alphabet} -self {file_train} -n {n} -r {r} -c -l < {file_test}'
    p = os.popen(cmd)
    return list(map(float, p.read().strip().split('\n')))  # parse as list of floats


def compute_scores(file_train, alphabet, file_test_self, file_test_anomaly, n, r):
    # compute the self and anomaly scores using the java program
    scores_self = compute_matches(file_train, file_test_self, alphabet, n, r)
    scores_anomaly = compute_matches(file_train, file_test_anomaly, alphabet, n, r)

    # label the scores as self/anomaly, i.e. false/true
    scores = np.array(
        [[score, False] for score in scores_self] +
        [[score,  True] for score in scores_anomaly]
    )

    # sort the scores
    return scores[scores[:, 0].argsort()], len(scores_self), len(scores_anomaly)


def compute_stats(scores, num_self, num_anomaly):
    # compute all sensitivity and specificity values in O(n)
    sensitivities = []
    specificities = []

    print(*scores, sep='\n')

    # Approach: compute sensitivity and specificity as normal, unless there are
    # multiple entries sharing the same score next to each other. If there are,
    # only compute the sensitivity and specificity after sweeping through to
    # the largest index of that "block" of same-score datapoints. This
    # reflects what happens in actual classification; a shift in the threshold
    # affects the outcome for all datapoints with that score at once. 

    count_anomaly = 0
    for cutoff_index in range(len(scores)):
        score, anomalous = scores[cutoff_index]
        
        if cutoff_index < len(scores) -1:
            next_score, _ = scores[cutoff_index + 1]
        else:
            next_score = -1

        if anomalous:
            count_anomaly += 1

        # Checking if we're currently at the end of the "block" of same-score
        # datapoints
        if score != next_score:
            sensitivities.append((num_anomaly - count_anomaly) / num_anomaly)
            specificities.append(1 - (cutoff_index - count_anomaly) / num_self)


    # add (0, 0) and (1, 1) and clip because of rounding errors
    sensitivities = np.clip([1] + sensitivities + [0], 0, 1)
    specificities = np.clip([1] + specificities + [0], 0, 1)

    # compute auc
    auc = sklearn.metrics.auc(specificities, sensitivities)
    return sensitivities, specificities, auc


def generate_plot(sensitivities, specificities, auc, file_train, file_test_self, file_test_anomaly, n, r):
    plt.figure(figsize=(5, 5))


    plt.plot(specificities, sensitivities, color='orange')
    plt.plot([0, 1], [0, 1], '--', color='#0e1111')

    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')

    if 'lang/' in file_test_anomaly:
        file_test_anomaly = file_test_anomaly[5:]

    plt.title(
        f'Trained on {file_train}, Tested on {file_test_self} and {file_test_anomaly}\n\
        Using n = {n} and r = {r}, AUC = {auc:.4f}'
    )
    plt.savefig(f'images/{file_train}_{file_test_self}_{file_test_anomaly}.png', bbox_inches="tight", dpi=300)
    plt.show()


def generate_9plot(results, file_train, file_test_self, file_test_anomaly, n):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    # hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # create the actual plots
    for index, (sensitivities, specificities, auc) in enumerate(results):
        axs[index // 3, index % 3].plot(specificities, sensitivities, color='orange')
        axs[index // 3, index % 3].plot([0, 1], [0, 1], '--', color='#0e1111')
        axs[index // 3, index % 3].set_title(f'r = {index + 1}, AUC = {auc:.4f}')

    # give labels two big labels
    fig.supxlabel('1 - specificity')
    fig.supylabel('sensitivity')

    # give on big title
    fig.suptitle(
        f'Trained on {file_train}, Tested on {file_test_self} and {file_test_anomaly}, n = {n}'
    )
    fig.tight_layout() # makes it less tight lol

    plt.savefig(f'images/9plot_{file_train}_{file_test_self}_{file_test_anomaly}.png', bbox_inches="tight", dpi=300)
    plt.show()


def compute_auc(file_train, alphabet, file_test_self, file_test_anomaly, n, r):
    scores, num_self, num_anomaly = compute_scores(
        file_train, alphabet, file_test_self, file_test_anomaly, n, r)
    sensitivities, specificities, auc = compute_stats(scores, num_self, num_anomaly)
    generate_plot(
        sensitivities, specificities, auc, file_train, file_test_self, file_test_anomaly, n, r)


def compute_aucs(file_train, alphabet, file_test_self, file_test_anomaly, n):
    results = []
    for r in range(1, 10):
        scores, num_self, num_anomaly = compute_scores(
            file_train, alphabet, file_test_self, file_test_anomaly, n, r)
        results.append(compute_stats(scores, num_self, num_anomaly))

    generate_9plot(results, file_train, file_test_self, file_test_anomaly, n)


def part_1(n, r):
    # Control: English vs Tagalog
    file_train = 'english.train'
    file_test_self = 'english.test'
    file_test_anomaly = 'tagalog.test'
    alphabet = 'file://english.train'
    compute_aucs(file_train, alphabet, file_test_self, file_test_anomaly, n)

    other_langs = ['lang/hiligaynon.txt', 
                   'lang/middle-english.txt',
                   'lang/plautdietsch.txt',
                   'lang/xhosa.txt']

    for lang in other_langs:
        compute_auc(file_train, alphabet, file_test_self, lang, n, r)




if __name__ == '__main__':
    part_1(10,3)
