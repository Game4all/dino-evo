import time
import pickle
from typing import Optional
import pygame
import numpy as np
import argparse

from tqdm import tqdm
from dino_evo.env import *
from dino_evo.agent import AgentBrain, relu, sigmoid, tanh, create_brain
from dino_evo.ga import ga_blx_alpha_crossover, ga_mutate, tournament_selection, HyperParamAnnealingSchedule, calculate_population_diversity
from dino_evo.vis import PyGameEnvVis

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Default population size for GA
POPULATION_SIZE = 200
# Default generation count for GA
NUM_GENERATIONS = 50
# Exploration factor for weights crossover aka alpha
ALPHA_FACTOR = 0.6
# Elitism factor
ELITISM_COUNT = 8
# Default seed
ENV_SEED = 0xDEAD

INITIAL_MUTATION_STRENGTH = 2e-1
FINAL_MUTATION_STRENGTH = 1e-5
INITIAL_MUTATION_RATE = 0.5
FINAL_MUTATION_RATE = 0.01

INPUT_NODES = 11
ACTIONS = list(DinoAction)

MIN_AIR_TIME_PENALTY = 0.0005
MAX_AIR_TIME_PENALTY = 0.5

AIR_TIME_PENALTY = 0.1

# def create_brain() -> AgentBrain:
#     """Creates a brain network to control a dino."""
#     return AgentBrain(INPUT_NODES, layer_configs=[
#         (8, relu),
#         (6, relu),
#         (len(DinoAction), tanh)
#     ])


def fitness_func(p, air_time_penalty) -> float:
    """Calculates the fitness score for a given player."""
    return p.score - p.total_air_time * air_time_penalty


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple NN agent using a genetic algorithm.")
    parser.add_argument('--train-gui', action='store_true',
                        help="Enable visualization of population during training using PyGame.")
    parser.add_argument('--showcase-best', action='store_true',
                        help="Visualize the best performing agent after training is complete.")
    parser.add_argument('-g', '--generations', type=int, default=NUM_GENERATIONS,
                        help=f"Number of generations to train for (default: {NUM_GENERATIONS}).")
    parser.add_argument('-p', '--population', type=int, default=POPULATION_SIZE,
                        help=f"Size of the population for each generation (default: {POPULATION_SIZE}).")
    parser.add_argument('--alpha', type=float, default=ALPHA_FACTOR,
                        help=f"Exploration factor for weight crossover between 0 and 1 aka alpha (default: {ALPHA_FACTOR}).")
    parser.add_argument('--air-penalty', type=float, default=AIR_TIME_PENALTY,
                        help=(f"Air-time penalty multiplier applied to total_air_time in the fitness function "
                              f"in range [{MIN_AIR_TIME_PENALTY}, {MAX_AIR_TIME_PENALTY}] (default: {AIR_TIME_PENALTY})."))
    parser.add_argument(
        '--save-to', type=str, default=None, help=f"The file to save the best model to. (optional)")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"The seed to use for RNG (defaults to {ENV_SEED})")
    args = parser.parse_args()

    # use arguments for population size and number of generations
    population_size = args.population
    num_generations = args.generations
    alpha = np.clip(args.alpha, 0, 1)
    seed = args.seed if args.seed else ENV_SEED
    air_time_penalty = np.clip(
        args.air_penalty, MIN_AIR_TIME_PENALTY, MAX_AIR_TIME_PENALTY)

    print(
        f"Starting training for {num_generations} generations with population size = {population_size}, "
        f"exploration factor (alpha) = {alpha}, air-time penalty = {air_time_penalty:.2f}, "
        f"seed = {seed}.")

    view, screen, clock, font = None, None, None, None
    if args.train_gui:
        pygame.init()
        pygame.display.set_caption("dino-evo training gui")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.Font("../assets/PressStart2P-Regular.ttf", 14)
        view = PyGameEnvVis(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)

    if args.train_gui:
        view.set_train_mode(True)
        print("GUI training mode enabled. Press 'T' to toggle turbo mode.")

    env = DinoRunnerEnv(population_size, seed, max_score=10_000)
    population = [create_brain(INPUT_NODES) for _ in range(population_size)]
    best_overall_score = -float('inf')
    best_overall_agent = None

    is_running = True
    turbo_mode = False

    hyperparam_schedule = HyperParamAnnealingSchedule(num_generations,
                                                      hyperparams={
                                                          "mutation_rate": (INITIAL_MUTATION_RATE, FINAL_MUTATION_RATE),
                                                          "mutation_strength": (INITIAL_MUTATION_STRENGTH, FINAL_MUTATION_STRENGTH),
                                                      })

    generation_iterator = range(num_generations)

    if not args.train_gui:
        generation_iterator = tqdm(
            generation_iterator, desc="Training Generations")

    # genetic algorithm main loop
    for generation in generation_iterator:
        # ===================== Evaluating the generation's population =====================
        observations = env.reset()
        if args.train_gui:
            view.update_train_info(generation + 1, num_generations)

        hyper_params = hyperparam_schedule.get_params(generation)
        generation_start_time = time.time()

        # inner loop for a single generation: run until all dinos are done
        while not env.is_done and is_running:
            if args.train_gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_t:
                            turbo_mode = not turbo_mode

            dino_actions = []
            for i, brain in enumerate(population):
                if not env.players[i].game_over:
                    obs = observations[i]
                    prediction = brain.predict(obs)
                    dino_actions.append(ACTIONS[np.argmax(prediction)])
                else:
                    dino_actions.append(DinoAction.NONE)

            observations = env.update(dino_actions)

            # draw the visualization if in train_gui training mode
            if args.train_gui:
                view.draw(env)
                pygame.display.flip()
                if turbo_mode:
                    clock.tick()
                else:
                    clock.tick(200)

        generation_end_time = time.time()

        if not is_running:
            break

        # ===================== Calculating fitness of each agent  =====================
        fitness_scores = [fitness_func(p, air_time_penalty)
                          for p in env.players]

        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        current_best_agent, current_best_score = population_with_fitness[0]
        if current_best_score > best_overall_score:
            best_overall_score = current_best_score
            best_overall_agent = current_best_agent

        avg_score = sum(fitness_scores) / len(fitness_scores)
        population_diversity = calculate_population_diversity(population)
        training_time = generation_end_time - generation_start_time

        log_message = (
            f"Gen {generation + 1} | Best fitness: {current_best_score:.2f} | Avg. fitness: {avg_score:.2f} | "
            f"Diversity: {population_diversity:.4f} | Sim duration: {training_time:.2f}s"
        )

        if args.train_gui:
            print(log_message)
        else:
            tqdm.write(log_message)

        # ===================== Creating a new population =====================
        new_population = []
        elites = [agent for agent,
                  score in population_with_fitness[:ELITISM_COUNT]]
        new_population.extend(elites)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population_with_fitness, k=5)
            parent2 = tournament_selection(population_with_fitness, k=5)

            while parent2 is parent1:
                parent2 = tournament_selection(population_with_fitness, k=5)

            parent1_weights = parent1.get_flattened_weights()
            parent2_weights = parent2.get_flattened_weights()

            child_weights = ga_blx_alpha_crossover(
                parent1_weights, parent2_weights, alpha, -2, 2)
            mutated_child_weights = ga_mutate(
                child_weights, hyper_params['mutation_rate'], hyper_params['mutation_strength'])

            child_nn = create_brain(INPUT_NODES)
            child_nn.set_flattened_weights(mutated_child_weights)
            new_population.append(child_nn)

        population = new_population

    # ===================== Visualizing the best overall agent =====================
    if best_overall_agent:
        print(
            f"\nTraining finished. Best fitness: {best_overall_score:.2f}")

        if args.save_to:
            pickle.dump(best_overall_agent.get_flattened_weights(),
                        open(args.save_to, "wb"))
            print(f"Saved model to pickled file: {args.save_to}")

        if not args.train_gui and args.showcase_best:
            pygame.init()
            pygame.display.set_caption("Best agent showcase_best")
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            clock = pygame.time.Clock()
            font = pygame.font.Font("../assets/PressStart2P-Regular.ttf", 14)
            view = PyGameEnvVis(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)

        if args.showcase_best:
            brains = [best_overall_agent]
            env = DinoRunnerEnv(len(brains), seed)
            view.set_train_mode(False)
            observation = env.reset()
            is_running = True

            while is_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                    if event.type == pygame.KEYDOWN and env.is_done:
                        observation = env.reset()

                if not env.is_done:
                    dino_actions = []
                    for (obs, brain) in zip(observation, brains):
                        actions = list(DinoAction)
                        next_action_predictions = brain.predict(obs)
                        dino_actions.append(
                            actions[np.argmax(next_action_predictions)])
                    observation = env.update(dino_actions)

                view.draw(env)
                pygame.display.flip()
                clock.tick(144)

    # deinitialize pygame if it was initialized
    if args.train_gui or args.showcase_best:
        pygame.quit()


if __name__ == "__main__":
    main()
