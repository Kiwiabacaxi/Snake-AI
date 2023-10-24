import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

import torch

# from torch.utils.tensorboard import SummaryWriter

pygame.init()
# font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont("arial", 25)


# oque tem que alterar do snake_human
# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision
# mandar o codigo pronto *


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 200

WIDTH = 400  # 640
HEIGHT = 400  # 480


class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT):
        # Seta o tamanho da tela
        self.w = w
        self.h = h

        # init log no Tensorboard
        # self.writer = SummaryWriter("logs")

        # init display do pygame
        self.display = pygame.display.set_mode(
            (self.w, self.h)
        )  # seta o tamanho da tela
        pygame.display.set_caption("Snake")  # seta o nome da tela
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init estado inicial do jogo
        self.direction = Direction.RIGHT  # começa indo para a direita

        # cria a cabeça da cobra como um ponto no meio da tela
        self.head = Point(self.w / 2, self.h / 2)

        # cria a cobra com 3 pontos
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        # inicia score como 0 e coloca a comida em um lugar aleatório
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        # log the score and food position - no tensorboard
        self.writer.add_scalar("score", self.score, self.frame_iteration)
        self.writer.add_scalar
        # return game state

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        # add reward and frame_iteration
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score

        # 7. log the score and food position
        self.writer.add_scalar("score", self.score, self.frame_iteration)
        self.writer.add_scalar("food_x", self.food.x, self.frame_iteration)
        self.writer.add_scalar("food_y", self.food.y, self.frame_iteration)
        self.writer.flush()
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
