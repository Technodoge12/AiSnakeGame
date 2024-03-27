import pygame
import random
pygame.init()
from collections import namedtuple
import numpy as np

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
block_size = 20
Point = namedtuple('Point', 'x, y')
font = pygame.font.SysFont('arial', 25)


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.speed = 200
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = 1

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - block_size, self.head.y),
                          Point(self.head.x - block_size * 2, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-block_size)//block_size)*block_size
        y = random.randint(0, (self.h - block_size) // block_size) * block_size
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        #1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.speed -= 100
                elif event.key == pygame.K_RIGHT:
                    self.speed += 100


        #2
        self._move(action)
        self.snake.insert(0, self.head)

        #3
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward += -10
            return reward, game_over, self.score

        #4
        if self.head == self.food:
            self.score += 1
            reward += 20
            self._place_food()
        else:
            self.snake.pop()
        #5
        self._update_ui()
        self.clock.tick(self.speed)
        #6
        game_over = False
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - block_size or pt.x < 0 or pt.y > self.h - block_size or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, block_size, block_size))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        text2 = font.render("Speed: " + str(self.speed), True, WHITE)
        self.display.blit(text2, [0, 100])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [1, 2, 3, 4]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] #right
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] #left
        self.direction = new_dir


        x = self.head.x
        y = self.head.y
        if self.direction == 1:
            x += block_size
        elif self.direction == 4:
            y -= block_size
        elif self.direction == 2:
            y += block_size
        elif self.direction == 3:
            x -= block_size
        self.head = Point(x, y)

        #D:\Code\Projects\AiSnake\model.py:36: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ..\torch\csrc\utils\tensor_new.cpp:278.)
  #state = torch.tensor(state, dtype=torch.float)