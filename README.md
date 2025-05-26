# Solving TSP Using Metaheuristics

The Traveling Salesman Problem (TSP) is a famous optimization challenge that seeks the shortest possible route visiting each city once and returning to the origin. Its importance lies in its NP-hard complexity and its wide range of applications in logistics, planning, and manufacturing. Due to the difficulty of finding optimal solutions for large datasets, metaheuristic algorithms like Genetic Algorithms, Simulated Annealing, Tabu Search, and Metropolis-Hastings can be used to obtain good approximate solutions efficiently. I compare these algorithms, whose peusdocode are outlined below.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         solving_tsp_using_metaheuristics and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── solving_tsp_using_metaheuristics   <- Source code for use in this project.
    │
    ├── metaheuristics
    │   ├── genetic_algorithm.py     <- Code executing genetic algorithm.
    │   ├── metropolis_hastings.py   <- Code executing metropolis hastings.
    │   ├── simulated_annealing.py   <- Code executing simulated annealing.
    │   ├── tabu_search.py           <- Code executing tabu search.
    │   └── utils.py                 <- Utility functions helpful for metaheuristic algorithms.
    │
    ├── __init__.py             <- Makes solving_tsp_using_metaheuristics a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │    
    └── plots.py                <- Code to create visualizations
```

--------

## Pseudocode

### Algorithm 1: Genetic Algorithm (Minimization)

**Require:** Randomly initialize population $$P_0$$, population size $$n$$, crossover rate $$\chi$$, mutation rate $$\mu$$, elitist rate $$\epsilon$$, maximum objective value $$\alpha$$.

1. $$k = 0$$  
2. **repeat**
   1. **Selection:** $$P' = \text{best } \epsilon\% \text{ of } P_k$$ to continue to new population.
   2. **Crossover:** Select top $$\chi\%$$ of $$P_k$$, pair them up, and produce offspring population $$O$$.
   3. **Mutation:** Select $$\mu\%$$ of solutions in $$O$$ and perturb a randomly-selected value in each solution. Then update $$P' = P' \cup O$$.
   4. **Evaluation:** Compute fitness $$f(x)$$ for all $$x \in P'$$.
   5. **Survival:** $$P_{k+1} = \text{top } N \text{ solutions from } P_k \cup P'$$ (minimizing $$f(x)$$).
3. Compute $$\mathbf{x}^* = \arg\min_{\mathbf{x} \in P_{k+1}} f(\mathbf{x})$$ and $$z^* = f(\mathbf{x}^*)$$.
4. Set $$k \leftarrow k+1$$.
5. **until** $$z^* < \alpha$$.
6. **return** $$\mathbf{x}^*, z^*$$ as the best solution found.

### Algorithm 2: Simulated Annealing

**Require:** Number of iterations $$N$$, cooling schedule $$\{T_k\}$$ such that $$\{T_k\} \to 0$$, number of Metropolis iterations $$M_k$$ for each temperature in $$\{T_k\}$$, and starting point $$\mathbf{x}^{(0)}$$.

1. Set $$z_0 = f(\mathbf{x}^{(0)})$$, $$\mathbf{x}^* = \mathbf{x}^{(0)}$$, and $$z^* = f(\mathbf{x}^*)$$.
2. **for** $$k = 0$$ to $$N$$ **do**
   1. **for** $$j = 0$$ to $$M_k$$ **do**
      1. Randomly select a neighbor $$\mathbf{x}'$$ and calculate $$z' = f(\mathbf{x}')$$.
      2. **if** $$z' < z_k$$ or $$\text{Unif}(0, 1) < \Pr(\mathbf{x}^{(k)} \to \mathbf{x}')$$ (using temperature $$T_k$$) **then**
         1. Set $$\mathbf{x}^{(k+1)} = \mathbf{x}'$$ and $$z_{k+1} = z'$$.
         2. **if** $$z_{k+1} < z^*$$ **then**
            1. Set $$z^* = z_{k+1}$$ and $$\mathbf{x}^* = \mathbf{x}^{(k+1)}$$.
         3. **end if**
      3. **else**
         1. Set $$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}$$ and $$z_{k+1} = z_k$$.
      4. **end if**
   2. **end for**
3. **end for**
4. **return** $$ \mathbf{x}^*, z^* $$ as the best solution found.

### Algorithm 3: Tabu Search (Minimization)

**Require:** Starting point $$\mathbf{x}^{(0)}$$, maximum number of iterations $$N$$, maximum tabu list size $$M$$, tabu rules set $$\text{tabuRules}()$$.

1. Set $$z_0 = f(\mathbf{x}^{(0)})$$, $$\mathbf{x}^* = \mathbf{x}^{(0)}$$, and $$z^* = f(\mathbf{x}^*)$$.
2. Set tabu list $$\mathcal{T} = \emptyset$$.
3. **for** $$k = 0$$ to $$N$$ **do**
   1. Compute neighborhood $$\mathcal{N}(\mathbf{x}^{(k)}) = \text{all neighbors of } \mathbf{x}^{(k)}$$.
   2. Remove tabu neighbors to obtain the allowable neighborhood: 
      $$\mathcal{N}^*(\mathbf{x}^{(k)}) = \mathcal{N}(\mathbf{x}^{(k)}) \setminus \mathcal{T}$$.
   3. Select best neighbor $$\mathbf{x}' \in \mathcal{N}^*(\mathbf{x}^{(k)})$$ and calculate $$z' = f(\mathbf{x}')$$.
   4. **if** $$z' < z^*$$ **then**
      1. Set $$z^* = z'$$ and $$\mathbf{x}^* = \mathbf{x}'$$.
   5. **end if**
   6. Update $$\mathcal{T}$$ according to the rules $$\text{tabuRules}()$$.
   7. **if** $$|\mathcal{T}| > M$$ **then**
      1. Remove the oldest $$|\mathcal{T}| - M$$ entries from $$\mathcal{T}$$.
   8. **end if**
4. **end for**
5. **return** $$\mathbf{x}^*, z^*$$ as the best solution found.

### Algorithm 4: Metropolis-Hastings (Minimization)

**Require:** Number of iterations $$N$$ and starting point $$\mathbf{x}^{(0)}$$.

1. Set $$z_0 = f(\mathbf{x}^{(0)})$$, $$\mathbf{x}^* = \mathbf{x}^{(0)}$$, and $$z^* = f(\mathbf{x}^*)$$.
2. **for** $$k = 0$$ to $$N$$ **do**
   1. Randomly select a neighbor $$\mathbf{x}'$$ and calculate $$z' = f(\mathbf{x}')$$.
   2. **if** $$z' < z_k$$ or $$\text{Unif}(0, 1) < \Pr(\mathbf{x}^{(k)} \to \mathbf{x}')$$ **then**
      1. Set $$\mathbf{x}^{(k+1)} = \mathbf{x}'$$ and $$z_{k+1} = z'$$.
      2. **if** $$z_{k+1} < z^*$$ **then**
         1. Set $$z^* = z_{k+1}$$ and $$\mathbf{x}^* = \mathbf{x}^{(k+1)}$$.
      3. **end if**
   3. **else**
      1. Set $$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}$$ and $$z_{k+1} = z_k$$.
   4. **end if**
3. **end for**
4. **return** $$\mathbf{x}^*, z^*$$ as the best solution found.

## Conclusions

The below depicts the consistent results for the most optimized route:
1. Tabu Search
2. Genetic Algorithm
3. Simulated Annealing
4. Metropolis-Hastings

## Acknowledgements

- Datasets obtained from https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html