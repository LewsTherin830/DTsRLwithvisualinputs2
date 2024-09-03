import numpy as np

class decision_boundary:
    
    def __init__(self, input_size):
        
        self.input_size = input_size
        self.W = np.random.uniform(low=-1, high=1, size=self.input_size)
        
    def set_params(self, W):
        self.W = W
    
    def get_output(self, inputs):
    
        
        eq = inputs@self.W.T
        
        if(eq < 0):
            return 0
        else:
            return 1
        
    def mutate(self):
        self.W = np.random.uniform(low=-1, high=1, size=self.input_size)
        
    def mutate_random_param(self):
        
        r = np.random.randint(0, self.input_size)
        
        if(r < self.input_size):
            self.W[r] = np.random.uniform(low=-1, high=1)
    
    def get_params(self):
        return self.W
    
    def copy(self):
        new = decision_boundary(len(self.W))
        new.set_params(self.W.copy())
        return new
        
        

class GeneticAlgorithmsDB:

    def __init__(self, input_size, pop_size, cx_prob, mut_prob, tournament_size):
        
        self._input_size = input_size
        self._pop_size = pop_size
        self._cx_prob = cx_prob
        self._mut_prob = mut_prob
        self._tournament_size = tournament_size
        self._pop = self._init_pop()


    def _init_pop(self):
        pop = []

        for i in range(self._pop_size):
            pop.append(decision_boundary(self._input_size))

        return pop

    def ask(self):
        return self._pop

    def _tournament_selection(self, fitnesses):
        n_ind = len(fitnesses)
        tournaments = np.random.choice(
            [*range(n_ind)],
            (n_ind, self._tournament_size)
        )

        selected = []

        for t in tournaments:
            max_ = float("-inf")
            argmax_ = None
            for idx in t:
                if fitnesses[idx] > max_ or argmax_ is None:
                    argmax_ = idx
                    max_ = fitnesses[idx]

            selected.append(argmax_)
        return selected

    def _crossover(self, par1, par2):
        p1, p2 = par1.copy(), par2.copy()
        
        W1 = p1.get_params()
        W2 = p2.get_params()
        
        crossover_point = np.random.randint(2, len(W1)-1)
    
        c1 = np.concatenate((W1[:crossover_point], W2[crossover_point:]))
        c2 = np.concatenate((W2[:crossover_point], W1[crossover_point:]))
        
        p1.set_params(c1)
        p2.set_params(c2)
        
        return p1,p2

    def _mutation(self, p):
        p1 = p.copy().mutate_random_param()

        return p1

    def tell(self, fitnesses):
        selection = self._tournament_selection(fitnesses)

        new_pop = []
        n_ind = len(selection)

        for i in range(0, n_ind, 2):
            p1 = self._pop[selection[i]]

            if i + 1 < n_ind:
                p2 = self._pop[selection[i + 1]]
            else:
                p2 = None

            o1, o2 = None, None

            # Crossover
            if p2 is not None and np.random.uniform() < self._cx_prob:
                o1, o2 = self._crossover(p1, p2)

            # Mutation
            if np.random.uniform() < self._mut_prob:
                o1 = self._mutation(p1 if o1 is None else o1)

            if p2 is not None and np.random.uniform() < self._mut_prob:
                o2 = self._mutation(p2 if o2 is None else o2)

            new_pop.append(p1 if o1 is None else o1)
            if p2 is not None:
                new_pop.append(p2 if o2 is None else o2)
        self._pop = new_pop

        
        
        