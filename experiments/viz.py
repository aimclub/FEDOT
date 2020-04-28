import matplotlib.pyplot as plt
import seaborn as sns


def show_fitness_by_generations(history_runs, iterations):
    iters = [it for it in range(iterations)]
    fitness_by_iter = []
    for it in iters:
        fitness_values = []
        for history in history_runs:
            value = abs(history[it])
            fitness_values.append(value)
        fitness_by_iter.append(fitness_values)

    sns.boxplot(iters, fitness_by_iter, color="seagreen")

    plt.title('Fitness history by generations')
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')
    plt.show()
