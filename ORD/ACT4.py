import numpy as np
import random
def fitness_function(chromosome, x_range):
    a, b, c = chromosome
    x_values = np.linspace(x_range[0], x_range[1], 100)
    y_values = a * x_values**2 + b * x_values + c
    return np.max(y_values)
def tournament_selection(population, fitness_values, tournament_size=3):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        selected_parent_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected_parents.append(population[selected_parent_index])
    return selected_parents
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2
def uniform_mutation(chromosome, mutation_rate=0.1):
    mutated_chromosome = chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] += random.uniform(-1, 1)
    return mutated_chromosome

# Parámetros del algoritmo genético
population_size = 100
chromosome_length = 3
x_range = (-10, 10)
mutation_rate = 0.1
generations = 100
population = np.random.uniform(-10, 10, size=(population_size, chromosome_length))

for generation in range(generations):
    fitness_values = [fitness_function(chromosome, x_range) for chromosome in population]
    selected_parents = tournament_selection(population, fitness_values)
    offspring = []
    for i in range(0, population_size, 2):
        child1, child2 = single_point_crossover(selected_parents[i], selected_parents[i+1])
        child1 = uniform_mutation(child1, mutation_rate)
        child2 = uniform_mutation(child2, mutation_rate)
        offspring.extend([child1, child2])
    population = np.array(offspring)
best_solution_index = np.argmax([fitness_function(chromosome, x_range) for chromosome in population])
best_solution = population[best_solution_index]
best_fitness = fitness_function(best_solution, x_range)

print("La mejor solución encontrada es:", best_solution)
print("Con un valor de aptitud de:", best_fitness)
