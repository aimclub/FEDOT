import matplotlib.pyplot as plt
import seaborn as sns


def show_fitness_by_generations(history_runs):
    gens = [gen for gen in range(len(history_runs[0]))]
    fitness_by_gens = []
    for gen in gens:
        fitness_values = []
        for history in history_runs:
            value = abs(history[gen][1])
            fitness_values.append(value)
        print(fitness_values)
        fitness_by_gens.append(fitness_values)

    sns.boxplot(gens, fitness_by_gens, color="seagreen")

    plt.title('Fitness history by generations')
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')
    plt.show()
