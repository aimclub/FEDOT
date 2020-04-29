from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns


def fitness_by_generations_boxplots(history_runs, iterations):
    iters = [it for it in range(iterations)]
    fitness_by_iter = []
    for it in iters:
        fitness_values = []
        for history in history_runs:
            value = history[it]
            fitness_values.append(value)
        fitness_by_iter.append(fitness_values)

    sns.boxplot(iters, fitness_by_iter, color="seagreen")

    plt.title('Fitness history by generations')
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')
    plt.show()


def show_fitness_history_all(history_runs, iterations):
    color_to_take = cycle('bgrcmykw')
    iters = [it for it in range(iterations)]

    for idx, history in enumerate(history_runs):
        sns.tsplot(history, iters, legend=True, color=next(color_to_take))

    plt.legend(labels=[idx for idx in range(len(history_runs))])
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')
    plt.show()
