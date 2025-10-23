import pygame
from .env import *

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GAME_FPS = 200

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DINO_COLOR = (83, 83, 83)
DEAD_DINO_COLOR = (180, 83, 83)  # Color for dinos that have collided
CACTUS_COLOR = (0, 150, 0)
BIRD_COLOR = (128, 128, 0)


class PyGameEnvVis:
    """
    A pygame-based visualizer for the the environment.
    """

    def __init__(self, screen, font, screen_width, screen_height):
        self.screen = screen
        self.font = font
        self.width = screen_width
        self.height = screen_height

        # Training related attributes
        self.training = False
        self.generation = 0
        self.max_generations = 0

    def set_train_mode(self, training: bool):
        """Sets the visualizer to training mode."""
        self.training = training

    def update_train_info(self, generation: int, max_generations: int):
        """Sets the visualizer to training mode."""
        self.generation = generation
        self.max_generations = max_generations

    def _unnormalize_rect(self, entity: BaseEntity) -> pygame.Rect:
        """Converts an entity's normalized rect into a pygame.Rect in pixels."""
        pixel_x = int(entity.x * self.width)
        pixel_y = int(entity.y * self.height)
        pixel_width = int(entity.width * self.width)
        pixel_height = int(entity.height * self.height)
        return pygame.Rect(pixel_x, pixel_y, pixel_width, pixel_height)

    def draw(self, game_model: DinoRunnerEnv):
        """Draws the entire game based on the current state of the model."""
        self.screen.fill(WHITE)

        # draw the ground
        ground_pixel_y = int(NORMALIZED_GROUND_Y * self.height)
        pygame.draw.rect(self.screen, GRAY, (0, ground_pixel_y,
                         self.width, self.height - ground_pixel_y))

        # draw all players
        for player in game_model.players:
            if player.game_over:
                continue

            player_rect = self._unnormalize_rect(player)
            pygame.draw.rect(self.screen, DINO_COLOR, player_rect)

        # draw all obstacles
        for obstacle_entity in game_model.obstacles:
            color = BIRD_COLOR if isinstance(
                obstacle_entity, PterodactylEntity) else CACTUS_COLOR
            obstacle_rect = self._unnormalize_rect(obstacle_entity)
            pygame.draw.rect(self.screen, color, obstacle_rect)

        offset_y = 10

        if self.training:
            training_text = self.font.render(
                "TRAINING", True, (255, 0, 0))
            self.screen.blit(training_text, (10, offset_y))
            offset_y += 30

            gen_text = self.font.render(
                f"Generation: {self.generation}/{self.max_generations}", True, BLACK)
            self.screen.blit(gen_text, (10, offset_y))
            offset_y += 20

        # display global game score
        score_text = self.font.render(
            f"Score: {int(game_model.game_progress_score)}", True, BLACK)
        self.screen.blit(score_text, (10, offset_y))

        offset_y += 20

        speed_text = self.font.render(
            f"Speed: {game_model.game_speed_percentage * 100.0:.2f}%", True, BLACK)
        self.screen.blit(speed_text, (10, offset_y))

        offset_y += 20

        # show how many agents are alive
        alive_dinos = sum(1 for p in game_model.players if not p.game_over)
        alive_text = self.font.render(
            f"Alive: {alive_dinos}/{game_model.num_dinos}", True, BLACK)
        self.screen.blit(alive_text, (10, offset_y))
