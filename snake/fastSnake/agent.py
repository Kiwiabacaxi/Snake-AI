import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 10000
LR = 0.001

# Começar aqui


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness || quantia de aleatoriedade
        self.gamma = 0.9  # discount rate || taxa de desconto

        # IA vai se lembrar de tudo que aconteceu no jogo
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() # if MAX_MEMORY is reached
        self.model = Linear_QNet(11, 256, 3)  # 11 inputs, 256 hidden layer, 3 outputs
        # self.model = Linear_QNet(11, 128, 3)  # 11 inputs, 128 hidden layer, 3 outputs
        self.trainer = QTrainer(
            self.model, lr=LR, gamma=self.gamma
        )  # trainer vai treinar o modelo

    def get_state(self, game):
        """Os 11 inputs são:
        1 - Danger straight - Perigo a frente
        2 - Danger right - Perigo a direita
        3 - Danger left - Perigo a esquerda

        4 - Direction LEFT - Direção da cobra para a esquerda
        5 - Direction RIGHT - Direção da cobra para a direita
        6 - Direction UP - Direção da cobra para cima
        7 - Direction DOWN - Direção da cobra para baixo

        8 - Food LEFT - Comida a esquerda
        9 - Food RIGHT - Comida a direita
        10 - Food UP - Comida acima
        11 - Food DOWN - Comida abaixo

        20 - 20 pixels é hardcoded msm :smugface:
        """

        head = game.snake[0]  # cabeça da cobra - lista de pontos
        point_l = Point(head.x - 20, head.y)  # ponto a esquerda da cabeça
        point_r = Point(head.x + 20, head.y)  # ponto a direita da cabeça
        point_u = Point(head.x, head.y - 20)  # ponto acima da cabeça
        point_d = Point(head.x, head.y + 20)  # ponto abaixo da cabeça

        dir_l = (
            game.direction == Direction.LEFT
        )  # se a direção da cobra for para a esquerda
        dir_r = (
            game.direction == Direction.RIGHT
        )  # se a direção da cobra for para a direita
        dir_u = game.direction == Direction.UP  # se a direção da cobra for para cima
        dir_d = game.direction == Direction.DOWN  # se a direção da cobra for para baixo

        state = [
            # Danger straight - se a cobra está indo para a esquerda e o ponto a esquerda da cabeça está na cobra
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right - se a cobra está indo para a direita e o ponto a direita da cabeça está na cobra
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left - se a cobra está indo para a esquerda e o ponto a esquerda da cabeça está na cobra
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction - se a cobra está indo para a esquerda
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location - se a comida está a esquerda da cabeça
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)  # retorna o estado da cobra

    def remember(self, state, action, reward, next_state, done):
        """Memorize the game
        Memoriza o jogo - partida imediatamente anterior"""
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        """Train the model with the memorized games
        Treina o modelo com os jogos memorizados
        """
        if len(self.memory) > BATCH_SIZE:
            # amostro aleatoriamente 10000 jogos
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # descompacta a lista de tuplas usando o zip - mais rapido que um for
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # ou faz pór loop de for
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_states, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model with the current game
        Treina o modelo com o jogo atual"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # nas fase iniciais do jogo, a cobra vai se mover aleatoriamente para explorar o ambiente
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # 80% -> 0%
        final_move = [0, 0, 0]
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
    # inicializa o treinamento/jogo
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state - estado anterior/atual
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state - realiza o movimento e pega o novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory - treina a memoria curta
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
