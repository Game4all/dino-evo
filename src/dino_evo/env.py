"""
Minimalist implementation of the chrome dino runner game
"""

from enum import Enum
import math
import random
from typing import List
import numpy as np

# =========== Game constants normalized in 1.0x1.0 space ==================

NORMALIZED_GROUND_Y = 0.9

DINO_STAND_WIDTH = 0.05
DINO_STAND_HEIGHT = 0.067
DINO_CROUCH_WIDTH = 0.075
DINO_CROUCH_HEIGHT = 0.045

DINO_INITIAL_X = 0.125
DINO_JUMP_SPEED = 0.03
DINO_GRAVITY = 0.001
DINO_FAST_FALL_GRAVITY = DINO_GRAVITY * 3

CACTUS_WIDTH = 0.03125
CACTUS_HEIGHT = 0.0625

PTERODACTYL_WIDTH = 0.05
PTERODACTYL_HEIGHT = 0.04
PTERODACTYL_Y_POSITIONS = [
    NORMALIZED_GROUND_Y - PTERODACTYL_HEIGHT - 0.01,
    NORMALIZED_GROUND_Y - PTERODACTYL_HEIGHT - 0.06,
    NORMALIZED_GROUND_Y - PTERODACTYL_HEIGHT - 0.11,
]

CACTUS_CONFIGS = [
    (CACTUS_WIDTH, CACTUS_HEIGHT),
    (CACTUS_WIDTH * 2, CACTUS_HEIGHT),
    (CACTUS_WIDTH * 3, CACTUS_HEIGHT),
]

INITIAL_GAME_SPEED = 0.00625
GAME_SPEED_INCREASE = 0.0000125
MAX_GAME_SPEED = 0.01875

# ================================ Base classes ===========================


class BaseEntity:
    """A base class for any game object that has a position, size, and hitbox."""

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def collides_with(self, other: 'BaseEntity') -> bool:
        """AABB collision detection."""
        self_right = self.x + self.width
        self_bottom = self.y + self.height

        other_right = other.x + other.width
        other_bottom = other.y + other.height

        if (self.x < other_right and self_right > other.x and
                self.y < other_bottom and self_bottom > other.y):
            return True
        return False


class DinoEntity(BaseEntity):
    """Holds the state and logic for the player in a normalized coordinate space."""

    def __init__(self):
        super().__init__(
            DINO_INITIAL_X, NORMALIZED_GROUND_Y -
            DINO_STAND_HEIGHT, DINO_STAND_WIDTH, DINO_STAND_HEIGHT
        )
        self.jump_speed = DINO_JUMP_SPEED
        self.gravity = DINO_GRAVITY
        self.velocity_y = 0.0
        self.is_jumping = False
        self.is_crouching = False
        self.is_fast_falling = False
        self.crouch_on_land = False

        # Dino-specific stats
        self.score = 0
        self.game_over = False

        # Behavioural stats
        self.total_air_time = 0  # total air time of the dino in ticks
        self.total_crouch_time = 0  # total crouch time of the dino in ticks

    def jump(self):
        """
        Makes the player jump.
        """
        if not self.is_jumping and not self.is_crouching:
            self.velocity_y = -self.jump_speed
            self.is_jumping = True
            self.crouch_on_land = False

    def crouch(self):
        """
        Makes the player crouch
        """
        if not self.is_jumping and not self.is_crouching:
            self.is_crouching = True
            self.width = DINO_CROUCH_WIDTH
            self.height = DINO_CROUCH_HEIGHT
            self.y = NORMALIZED_GROUND_Y - self.height

    def stand_up(self):
        """
        Makes the player stand up after crouching.
        """
        self.crouch_on_land = False
        if self.is_crouching:
            self.is_crouching = False
            self.width = DINO_STAND_WIDTH
            self.height = DINO_STAND_HEIGHT
            self.y = NORMALIZED_GROUND_Y - self.height

    def fast_fall(self):
        """
        Makes the player fall faster to the ground.
        """
        if self.is_jumping:
            self.is_fast_falling = True
            self.crouch_on_land = True

    def update(self):
        """
        Updates the player physics.
        """
        if self.is_fast_falling:
            self.velocity_y += DINO_FAST_FALL_GRAVITY
        else:
            self.velocity_y += self.gravity

        self.y += self.velocity_y

        if self.is_jumping:
            self.total_air_time += 1

        if self.is_crouching:
            self.total_crouch_time += 1

        if self.y + self.height >= NORMALIZED_GROUND_Y:
            self.is_jumping = False
            self.is_fast_falling = False
            self.velocity_y = 0

            if self.crouch_on_land:
                self.is_crouching = True
                self.width = DINO_CROUCH_WIDTH
                self.height = DINO_CROUCH_HEIGHT
                self.crouch_on_land = False

            self.y = NORMALIZED_GROUND_Y - self.height


class CactusEntity(BaseEntity):
    def __init__(self, speed, rng: random.Random, width, height):
        spawn_offset = rng.uniform(0.0, 0.375)
        x = 1.0 + spawn_offset
        y = NORMALIZED_GROUND_Y - height
        super().__init__(x, y, width, height)
        self.speed = speed

    def update(self):
        self.x -= self.speed


class PterodactylEntity(BaseEntity):
    def __init__(self, speed, rng: random.Random):
        spawn_offset = rng.uniform(0.0, 0.375)
        x = 1.0 + spawn_offset
        y = rng.choice(PTERODACTYL_Y_POSITIONS)
        super().__init__(x, y, PTERODACTYL_WIDTH, PTERODACTYL_HEIGHT)
        self.speed = speed

    def update(self):
        self.x -= self.speed


class DinoAction(Enum):
    NONE = 0
    JUMP = 1
    CROUCH = 2


class DinoRunnerEnv:
    def __init__(self, num_dinos: int = 1, seed=0xDEADCAFE, max_score: int | None = None):
        self.num_dinos = num_dinos
        self.rng_seed = seed
        self.rng = random.Random(seed)
        self.max_score = max_score
        self.reset()

    def reset(self) -> np.ndarray:
        self.players = [DinoEntity() for _ in range(self.num_dinos)]
        self.obstacles = []
        self.game_speed = INITIAL_GAME_SPEED
        self.game_progress_score = 0
        self.steps = 0
        self.obstacle_timer = 0
        self.rng = random.Random(self.rng_seed)

        return self.get_env_state()

    def update(self, actions: List[DinoAction]) -> np.ndarray:
        if self.is_done:
            return self.get_env_state()

        for player, action in zip(self.players, actions):
            if player.game_over:
                continue

            match action:
                case DinoAction.JUMP:
                    player.jump()
                case DinoAction.CROUCH:
                    if player.is_jumping:
                        player.fast_fall()
                    else:
                        player.crouch()
                case DinoAction.NONE:
                    player.stand_up()

            player.update()

        for obstacle in self.obstacles:
            obstacle.update()

        self.obstacles = [
            obs for obs in self.obstacles if obs.x + obs.width > 0]

        # update game progress if at least one dino is alive.
        if any(not p.game_over for p in self.players):
            self.game_progress_score += 0.1

            # increase game speed at intervals
            if int(self.game_progress_score) % 50 == 0 and self.game_progress_score > 1:
                self.game_speed += GAME_SPEED_INCREASE
                self.game_speed = min(MAX_GAME_SPEED, self.game_speed)
                for obstacle in self.obstacles:
                    obstacle.speed = self.game_speed

            # obstacle spawning
            if not self.obstacles or self.obstacle_timer <= 0:
                can_spawn_pterodactyl = self.game_progress_score > 500

                if can_spawn_pterodactyl and self.rng.random() < 0.3:
                    self.obstacles.append(
                        PterodactylEntity(self.game_speed, self.rng))
                else:
                    width, height = self.rng.choice(CACTUS_CONFIGS)
                    self.obstacles.append(CactusEntity(
                        self.game_speed, self.rng, width, height))

                self.obstacle_timer = self.rng.randint(100, 300)

            self.obstacle_timer -= 1

        # check for collisions and update scores
        for player in self.players:
            if player.game_over:
                continue

            # increment score for active players
            player.score += 0.1

            # checking for collision
            for obstacle in self.obstacles:
                if player.collides_with(obstacle):
                    player.game_over = True
                    break

            if player.game_over:
                continue

            if self.max_score and player.score > self.max_score:
                player.game_over = True

        self.steps += 1
        return self.get_env_state()

    def _get_state_for_dino(self, player: DinoEntity) -> np.ndarray:
        """Returns a ndarray representing the environment state for a single dino."""
        player_y = player.y
        player_y_velocity = player.velocity_y
        player_is_fastfalling = 1.0 if player.is_fast_falling else 0.0
        player_is_jumping = 1.0 if player.is_jumping else 0.0

        upcoming_obstacles = [
            obs for obs in self.obstacles if obs.x > player.x]

        if len(upcoming_obstacles) > 0:
            obs_1 = upcoming_obstacles[0]
            obs_1_x_dist = obs_1.x - (player.x + player.width)
            obs_1_y = obs_1.y
            obs_1_width = obs_1.width
        else:
            obs_1_x_dist, obs_1_y, obs_1_width = 1.0, 1.0, 0.0

        if len(upcoming_obstacles) > 1:
            obs_2 = upcoming_obstacles[1]
            obs_2_x_dist = obs_2.x - (player.x + player.width)
            obs_2_y = obs_2.y
            obs_2_width = obs_2.width
        else:
            obs_2_x_dist, obs_2_y, obs_2_width = 1.0, 1.0, 0.0

        return np.array([
            player_y,
            player_y_velocity,
            player_is_jumping,
            player_is_fastfalling,
            self.game_speed,
            obs_1_x_dist,
            obs_1_y,
            obs_1_width,
            obs_2_x_dist,
            obs_2_y,
            obs_2_width
        ])

    def get_env_state(self) -> np.ndarray:
        """Returns a ndarray representing the environment state for all dinos."""
        states = [self._get_state_for_dino(p) for p in self.players]
        return np.array(states)

    @property
    def is_done(self) -> bool:
        """Checks if all dinos are KOed."""
        return all(player.game_over for player in self.players)

    @property
    def game_speed_percentage(self) -> float:
        """
        Returns the game speed as a percentage for display.
        """
        return self.game_speed / MAX_GAME_SPEED
