import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics


def process_training_data(n, dataset_directory):
    # Splits the training data for part 2 of the assignment into non-overlapping 
    # chunks of a fixed size. 

    # Read the lines of the file and store them in a list
    with open(dataset_directory, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    # Iterate through each line and split it into chunks of chunk_length
    chunked_data = []
    for line in lines:
        if len(line) < n: # Skipping if shorter than the length
            continue
        elif len(line) == n:
            chunked_data.append(line)
        else:
            # Chunking the string, ignoring the remnants at the very end 
            # if the chunk is smaller than the set length
            line_chunks = []
            for i in range(0, len(line), n):
                chunk = line[i:i+n]
                if len(chunk) == n:
                    line_chunks.append(chunk)

            # Adding chunks of the line to the complete set
            chunked_data += line_chunks

    # Removing duplicate chunks
    chunked_data = np.unique(np.array(chunked_data))

    return chunked_data

def process_testing_data(n, dataset_directory):
    # Splits the testing data for part 2 of the assignment into non-overlapping
    # chunks of fixed size, keeping track of which chunks belong to which
    # original string for computing composite anomaly scores

    # Read the lines of the data file and store them in a list
    with open(dataset_directory, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    # Iterate through each line and split it into chunks of chunk_length
    chunked_data = []
    chunk_id = []
    for i, line in enumerate(lines):
        if len(line) < n: # Skipping if shorter than the length
            continue
        elif len(line) == n:
            chunked_data.append(line)
            chunk_id.append(i)
        else:
            # Chunking the string, ignoring the remnants at the very end 
            # if the chunk is smaller than the set length
            line_chunks = []
            for j in range(0, len(line), n):
                chunk = line[j:j+n]
                if len(chunk) == n:
                    line_chunks.append(chunk)

            # Adding chunks of the line to the complete set
            chunked_data += line_chunks
            # Adding the ID as many times as there are chunks
            chunk_id += (np.ones(len(line_chunks))*i).astype(int).tolist()

    return (chunked_data, chunk_id)


def compute_matches(file_train, file_test, alphabet, n, r):
    # train using the given training file, and compute the scores for the given test file
    cmd = f'java -jar negsel2.jar -alphabet {alphabet} -self {file_train} -n {n} -r {r} -c -l < {file_test}'
    p = os.popen(cmd)
    return np.array(list(map(float, p.read().strip().split('\n')))) # parse as list of floats


def compute_scores(file_train, file_test, alphabet, labels, chunk_ids, n, r):
    # training + testing using the java program
    scores = compute_matches(file_train, file_test, alphabet, n, r)
    # computing composite anomaly scores for the complete strings,
    # where chunk_ids indicate which string the chunk is originally from
    all_ids = np.unique(chunk_ids)
    composite_scores = np.zeros(len(all_ids))
    for i, chunk_id in enumerate(all_ids):
        composite_score = np.mean(scores[np.argwhere(chunk_ids == chunk_id)])
        composite_scores[i] = composite_score

    # finding the labels of the original strings which were used (some may
    # have been discarded due to being too short)
    used_labels = labels[np.sort(all_ids)]
    # merging the scores with the corresponding labels in one array
    scores = np.stack((composite_scores, used_labels), axis=1)
    num_anomaly = np.sum(scores, axis=0)[1]
    num_self = scores.shape[0]-num_anomaly    

    # sort the scores
    return scores[scores[:, 0].argsort()], num_self, num_anomaly


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


def generate_9plot(results, file_train, file_test, n):
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
        f'Trained on {file_train}, Tested on {file_test}, n = {n}'
    )
    fig.tight_layout() # makes it less tight lol

    plt.savefig(f'images/9plot_{file_train[17:]}_{file_test[17:]}.png', bbox_inches="tight", dpi=300)
    plt.show()

def compute_aucs(file_train, alphabet, file_test, labels, chunk_ids, n):
    results = []
    for r in range(1, 10):
        scores, num_self, num_anomaly = compute_scores(
            file_train, file_test, alphabet, labels, chunk_ids, n, r)
        results.append(compute_stats(scores, num_self, num_anomaly))

    generate_9plot(results, file_train, file_test, n)


def part_2(n):

    # Pre-processing the training data by chunking it
    cert_train_chunked = process_training_data(
        n, 'syscalls/snd-cert/snd-cert.train')
    unm_train_chunked = process_training_data(
        n, 'syscalls/snd-unm/snd-unm.train')

    # Iterating through all of the testing data stored in 
    # all of the different files, and chunking it too
    cert_test_chunked = []
    cert_test_chunk_ids = []
    cert_test_labels = []
    unm_test_chunked = []
    unm_test_labels = []
    unm_test_chunk_ids = []
    # Needed to correctly assign chunk indices to the merged data 
    previous_highest_index_cert = 0     
    previous_highest_index_unm = 0

    for i in range(1,4):
        cert_data, cert_chunk_ids = process_testing_data(
            n, f'syscalls/snd-cert/snd-cert.{i}.test')
        unm_data, unm_chunk_ids = process_testing_data(
            n, f'syscalls/snd-unm/snd-unm.{i}.test')  

        # Aggregating the labels for the COMPLETE strings
        # in two files while we're at it
        with open(f'syscalls/snd-unm/snd-unm.{i}.labels', 'r') as file:
            unm_labels = file.readlines()
        unm_labels = [int(line.strip()) for line in unm_labels] 

        with open(f'syscalls/snd-cert/snd-cert.{i}.labels', 'r') as file:
            cert_labels = file.readlines()
        cert_labels = [int(line.strip()) for line in cert_labels]


        cert_test_chunked += cert_data
        cert_test_labels += cert_labels 
        cert_test_chunk_ids += (np.array(cert_chunk_ids) + previous_highest_index_cert).tolist()
        unm_test_chunked += unm_data 
        unm_test_labels += unm_labels 
        unm_test_chunk_ids += (np.array(unm_chunk_ids) + previous_highest_index_unm).tolist()

        previous_highest_index_cert = np.max(cert_test_chunk_ids) + 1
        previous_highest_index_unm = np.max(unm_test_chunk_ids) + 1

    cert_test_labels = np.array(cert_test_labels)
    cert_test_chunk_ids = np.array(cert_test_chunk_ids)
    unm_test_labels = np.array(unm_test_labels)
    unm_test_chunk_ids = np.array(unm_test_chunk_ids)

    # Saving training data
    with open('syscalls/snd-cert/snd-cert_chunked.train', 'w') as file:
        for chunk in cert_train_chunked:
            file.write(str(chunk) + '\n')

    with open('syscalls/snd-unm/snd-unm_chunked.train', 'w') as file:
        for chunk in unm_train_chunked:
            file.write(str(chunk) + '\n')

    # Saving test data
    with open('syscalls/snd-cert/snd-cert_chunked.test', 'w') as file:
        for chunk in cert_test_chunked:
            file.write(str(chunk) + '\n')

    with open('syscalls/snd-unm/snd-unm_chunked.test', 'w') as file:
        for chunk in unm_test_chunked:
            file.write(str(chunk) + '\n')

    # Training
    alphabet_cert = 'file://syscalls/snd-cert/snd-cert.alpha'
    alphabet_unm = 'file://syscalls/snd-unm/snd-unm.alpha'


    compute_aucs('syscalls/snd-cert/snd-cert_chunked.train', alphabet_cert, 
        'syscalls/snd-cert/snd-cert_chunked.test', 
        cert_test_labels, cert_test_chunk_ids, n)

    compute_aucs('syscalls/snd-unm/snd-unm_chunked.train', alphabet_unm, 
        'syscalls/snd-unm/snd-unm_chunked.test', 
        unm_test_labels, unm_test_chunk_ids, n)
 
if __name__ == '__main__':
    part_2(10)

