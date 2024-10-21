import random
import math

def genetic_algorithm(users, GRID_SIZE, NUM_USERS, NUM_ANTENNAS, COVERAGE_RADIUS):
    # Algorithm-specific parameters
    POPULATION_SIZE = 50
    GENERATIONS = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1

    # Fitness function
    def fitness(individual):
        total_coverage = 0
        total_interference = 0
        covered_users = set()

        for user_idx, user in enumerate(users):
            signals = []
            for antenna in individual:
                distance = math.hypot(user[0] - antenna[0], user[1] - antenna[1])
                if distance <= COVERAGE_RADIUS:
                    signal_strength = 1 / (1 + distance)
                    signals.append(signal_strength)

            if signals:
                total_coverage += max(signals)
                if len(signals) > 1:
                    total_interference += sum(signals) - max(signals)
                covered_users.add(user_idx)

        coverage_score = len(covered_users) / NUM_USERS
        fitness_value = coverage_score - 0.1 * total_interference
        return fitness_value

    # Create initial population
    def create_population():
        population = []
        for _ in range(POPULATION_SIZE):
            individual = [
                (random.uniform(0, GRID_SIZE), random.uniform(0, GRID_SIZE))
                for _ in range(NUM_ANTENNAS)
            ]
            population.append(individual)
        return population

    # Selection function
    def selection(population, fitnesses):
        selected = []
        for _ in range(POPULATION_SIZE):
            i, j = random.sample(range(POPULATION_SIZE), 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            selected.append(winner)
        return selected

    # Crossover function
    def crossover(parent1, parent2):
        if random.random() < CROSSOVER_RATE:
            point = random.randint(1, NUM_ANTENNAS - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return [child1, child2]
        else:
            return [parent1.copy(), parent2.copy()]

    # Mutation function
    def mutate(individual):
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, NUM_ANTENNAS - 1)
            individual[idx] = (random.uniform(0, GRID_SIZE), random.uniform(0, GRID_SIZE))
        return individual

    # Initialize population
    population = create_population()
    best_fitness_history = []
    best_individual = None
    best_fitness = float('-inf')

    # Evolution loop
    for generation in range(GENERATIONS):
        fitnesses = [fitness(individual) for individual in population]
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[fitnesses.index(max_fitness)]
        best_fitness_history.append(best_fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")

        # Selection
        selected = selection(population, fitnesses)

        # Crossover and Mutation
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < POPULATION_SIZE else selected[0]
            offspring = crossover(parent1, parent2)
            next_generation.extend([mutate(child) for child in offspring])

        population = next_generation[:POPULATION_SIZE]

    return best_individual, best_fitness_history
