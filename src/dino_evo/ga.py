import random
import numpy as np


# ===================================== Mutation + crossover operators =====================================

def ga_single_point_crossover(parent1_weights: np.ndarray, parent2_weights: np.ndarray) -> np.ndarray:
    """Performs single-point crossover between two parent chromosomes."""

    child_weights = np.copy(parent1_weights)
    crossover_point = random.randint(1, len(child_weights) - 2)
    child_weights[crossover_point:] = parent2_weights[crossover_point:]

    return child_weights


def ga_blx_alpha_crossover(parent1_weights: np.ndarray, parent2_weights: np.ndarray, alpha, lower_bound, upper_bound) -> np.ndarray:
    """Performs BLX-alpha crossover between two parent chromosomes."""

    gene_range = np.abs(parent1_weights - parent2_weights)
    c_min = np.minimum(parent1_weights, parent2_weights)
    c_max = np.maximum(parent1_weights, parent2_weights)

    new_lower_bound = c_min - alpha * gene_range
    new_upper_bound = c_max + alpha * gene_range

    # generate new genes for a single offspring
    offspring_genes = np.random.uniform(
        new_lower_bound, new_upper_bound, size=parent1_weights.shape)
    # clip the generated offspring in the lower and upper bounds
    offspring = np.clip(offspring_genes, lower_bound, upper_bound)

    return offspring


def ga_mutate(weights: np.ndarray, mutation_rate: float, mutation_strength: float) -> np.ndarray:
    """Randomly mutates the genes (weights) of a chromosome."""
    mutated_weights = np.copy(weights)
    for i in range(len(mutated_weights)):
        if random.random() < mutation_rate:
            mutation = np.random.randn() * mutation_strength
            mutated_weights[i] += mutation
    return mutated_weights


def tournament_selection(population: list[tuple[object, float]], k: int = 4):
    """Perform tournament selection on the population."""
    tournament_contestants = random.sample(population, k)
    winner = max(tournament_contestants, key=lambda item: item[1])
    return winner[0]

# ====================================== Diversity tracking ======================================


def cosine_dist(vec1, vec2):
    """
    Calculates the cosine distance between two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0

    cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return 1 - cosine_similarity


def calculate_population_diversity(population: list):
    """
    Calculates the average cosine distance of each agent from the population's mean weight vector.
    """
    if len(population) < 2:
        return 0.0

    all_weights = [agent.get_flattened_weights() for agent in population]
    mean_weights = np.mean(all_weights, axis=0)

    total_distance = 0.0
    for weights in all_weights:
        total_distance += cosine_dist(weights, mean_weights)

    return total_distance / len(population)


class HyperParamAnnealingSchedule:
    """
    A class to manage annealing schedules for hyperparameters over generations for the GA.
    """

    def __init__(self, num_generations: int, hyperparams: dict):
        """
        Initializes the annealing schedule.
        Args:
            num_generations (int): total number of generations/steps in the GA run.
            hyperparams (dict): A dictionary of hyper params that should follow an annealing schedule in the form of (initial, final) value pair.
        """

        if not isinstance(num_generations, int) or num_generations < 1:
            raise ValueError("num_generations must be a positive")

        if not isinstance(hyperparams, dict) or not hyperparams:
            raise ValueError("hyperparams must be a non-empty dictionary.")

        self.n_generations = num_generations
        self.hyperparams = hyperparams

    def get_params(self, generation: int) -> dict:
        """
        Calculates the current value for each hyperparameter at a given generation.
        Args:
            generation (int): The current generation number (0-indexed).
        Returns:
            dict: a dict with the current values of the hyperparameters.
        """

        if not (0 <= generation < self.n_generations):
            raise ValueError(
                f"generation must be between 0 and {self.n_generations - 1}")

        # calculate progress ratio
        progress = (generation / (self.n_generations - 1)
                    if self.n_generations > 1 else 1.0)

        current_params = {}
        for name, (initial_val, final_val) in self.hyperparams.items():
            # apply the annealing formula current = initial * (final / initial) ^ progress
            # handle case where initial_val is 0 to avoid division by zero
            if initial_val == 0:
                current_value = final_val * progress
            else:
                ratio = final_val / initial_val
                current_value = initial_val * (ratio ** progress)

            current_params[name] = current_value

        return current_params
