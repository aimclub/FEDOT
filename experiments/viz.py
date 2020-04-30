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


def show_fitness_history_all(history_runs, iterations, with_bands=False):
    color_to_take = cycle('bgrcmykw')
    iters = [it for it in range(iterations)]

    if not with_bands:
        for history in history_runs:
            sns.tsplot(history, iters, legend=True, color=next(color_to_take))
        plt.legend(labels=[idx for idx in range(len(history_runs))])
    else:
        sns.tsplot(history_runs, iters, legend=True, color=next(color_to_take))
    plt.ylabel('Fitness')
    plt.xlabel('Iteration, #')
    plt.show()


def show_history_optimization_comparison(first, second, iterations,
                                         label_first, label_second):
    color_to_take = cycle('bgrcmykw')
    iters = [it for it in range(iterations)]

    sns.tsplot(first, iters, legend=True, color=next(color_to_take))
    sns.tsplot(second, iters, legend=True, color=next(color_to_take))
    plt.legend(labels=[label_first, label_second])

    plt.ylabel('Fitness')
    plt.xlabel('Iteration, #')
    plt.show()
