import argparse
import pickle
import pygame
import numpy as np

from dino_evo.env import DinoRunnerEnv, DinoAction
from dino_evo.agent import AgentBrain, relu, tanh
from dino_evo.vis import PyGameEnvVis

# Display settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ENV_SEED = 0xDEAD

INPUT_NODES = 11  # same as original
ACTIONS = list(DinoAction)

def create_brain() -> AgentBrain:
    """Recreate the same network used during training."""
    return AgentBrain(INPUT_NODES, layer_configs=[
        (8, relu),
        (6, relu),
        (len(DinoAction), tanh)
    ])


def load_trained_brain(model_path: str) -> AgentBrain:
    """Load and return model weights into a fresh brain."""
    weights = pickle.load(open(model_path, "rb"))
    brain = create_brain()
    brain.set_flattened_weights(weights)
    return brain


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained Dino-Evo agent.")
    parser.add_argument("--model", type=str,
                        help="Path to saved model .pkl file to visualize", required=True)
    parser.add_argument("--seed", type=int, default=ENV_SEED,
                        help=f"Seed to use for RNG (defaults to {ENV_SEED})")
    args = parser.parse_args()


    print(f"Loading model from: {args.model}, with seed = {args.seed}")
    brain = load_trained_brain(args.model)

    # init py game
    pygame.init()
    pygame.display.set_caption("dino-evo agent visualization")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font("../assets/PressStart2P-Regular.ttf", 14)
    view = PyGameEnvVis(screen, font, SCREEN_WIDTH, SCREEN_HEIGHT)

    env = DinoRunnerEnv(1, args.seed)
    view.set_train_mode(False)

    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and env.is_done:
                obs = env.reset()

        if not env.is_done:
            actions = []
            for o in obs:
                pred = brain.predict(o)
                actions.append(ACTIONS[np.argmax(pred)])
            obs = env.update(actions)

        view.draw(env)
        pygame.display.flip()
        clock.tick(144)

    pygame.quit()


if __name__ == "__main__":
    main()
