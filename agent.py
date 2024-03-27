import torch
import random
import numpy as np
from main import SnakeGame, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() when exeedes memory
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # model, trainer

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_r = game.direction == 1
        dir_d = game.direction == 2
        dir_l = game.direction == 3
        dir_u = game.direction == 4

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)),

            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)),

            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.x > game.head.x,
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #popleft if max mem is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for states, actions, rewards, next_states, dones in mini_sample:
            #self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation. you value randomness at the start as it helps for learning, but later its better to exploit the value of the agent
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)





if __name__ == '__main__':
    train()