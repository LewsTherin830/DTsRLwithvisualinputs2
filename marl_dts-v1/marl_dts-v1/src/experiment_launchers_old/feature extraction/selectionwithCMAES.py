from time import time
import random
import numpy as np
import continuous_optimization
from PIL import Image



#global_var
#initializing the dataset
dataset = np.load('resized.npy')
noise = np.load('noisy.npy')


#query variables
query_width = 5
no_queries = 1
stride = 1


#training variables
pop_size = 10
generations = 10


def show_im(q1):
    
    img = Image.fromarray(q1.reshape(5,5,3), 'RGB')
    img.show()


def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a, axis=1) * np.linalg.norm(b))
    
    
def attention(observation, queries):
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """
    
    h, w, _ = observation.shape
    delta = query_width
    img_patches = []
    features = []
    
    # q = queries.reshape(3,5,5,3)[0]
    # print(q[:,:,0])
    # from matplotlib import pyplot as plt
    # plt.imshow(res, interpolation='nearest')
    # plt.show()

    for j in range(0, h - delta, stride):
        for i in range(0, w - delta, stride):
            img_patches.append(observation[j:j+delta, i:i+delta].flatten())
    img_patches = np.array(img_patches)

    w_locations = np.floor((w - delta) / stride)
    
    # print(queries.shape)
    
    for q in queries:
        
        
        attention_scores = cosine_similarity(img_patches, q)
        best = np.argmax(attention_scores)

        x = best % w_locations
        y = best // w_locations

        features.extend([x, y])
    return features


def evaluate(queries):
    
    f1 = []
    fn = []
    
    for obs in dataset:
        f = attention(obs, queries)
        f1.append(f)
        
    f1 = np.array(f1)
    
    print(f1[:,0])
        
    for obs in noise:
        f = attention(obs, queries)
        fn.append(f)
        
    fn = np.array(fn)
    
    fitness =  max(np.std(f1[:,0]),np.std(f1[:,1])) - np.sum(abs(fn-f1))/13 #- abs(np.sum(abs(queries)))

    return fitness




if __name__ == "__main__":
    
    
    
    
        # "query_width": 5,
        # "stride": 3,
        # "n_queries": 3
        
        # "algorithm": "CMAES",
        # "args": {
        #     "lambda_": 2
        # }
        
    
    # Variables
    
    # Initialize continuous optimization algorithm
    
    query_len = query_width ** 2 * 3
    n_params = query_len * no_queries
    co = getattr(continuous_optimization, "CMAES")(n_params, lambda_ = pop_size)

    # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False

    # with open(os.path.join(log_path, "log.txt"), "a") as f:
    #     f.write(f"Generation Min Mean Max Std Invalid Time\n")
    print(
        f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
        {'Max': <10} {'Std': <10} {'Time': <10}"
    )

    # Iterate over the generations
    for gen in range(generations):

        # Retrieve the current population of queries
        qpop = co.ask()
        
        
        #calculate fitnesses
        fitnesses = []
        
        for q in qpop:
            queries = np.array(q).reshape((-1, query_len))
            fitness = evaluate(queries)
            fitnesses.append(fitness)
            
            print(fitness)
            
        ind_max = np.argmax(fitnesses)
        max_ = fitnesses[ind_max]
        
        # Check whether the best has to be updated
        if max_ > best_fit:
            best = qpop[ind_max]
            best_fit = max_
            new_best = True


        # Tell the fitnesses to CO
        co.tell(fitnesses)

        # Compute stats
        min_ = np.min(fitnesses)
        mean = np.mean(fitnesses)
        max_ = np.max(fitnesses)
        std = np.std(fitnesses)
        cur_t = time()

        print(
            f"{gen: <10} {min_: <10.2f} {mean: <10.2f} \
            {max_: <10.2f} {std: <10.2f} {cur_t: <10}"
        )

        # Update the log file
        # with open(os.path.join(log_path, "log.txt"), "a") as f:
        #     f.write(f"{gen} {min_} {mean} {max_} {std} {cur_t}\n")
        # #     if new_best:
        # #         f.write(f"New best pair.\nTree: {best_tree}; \
        # #                 Queries: {best_queries}; Fitness: {best_fit}\n")
        
        new_best = False
        
        
    np.save("kernel", np.array(best))


