import torch
import random
import numpy as np
from collections import deque #data structure to store "memories"
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """
    Training Function:
        - state = get_state(game)
        - action = get_move(state):
            model.predict()
        - reward, game_over, score = game.play_step(action)
        - new_state = get_state(game)
        - remember
        - model.train()
    """
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # randomness control parameter
        self.gamma = 0.9      # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if maxlen is exceeded
        self.model = Linear_QNet(11, 256, 3) # 11 state values, 256 hidden nodes, 3 output states...
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        11 values
        [   0,0,0          -> danger straight, danger right, danger left,
            0,1,0,0        -> direction left, right, up, down
            0,1,0,1   ]    -> food left, right, up, down
        """
        head = game.snake[0]
        # create points next to head in all directions...
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # boolean to check direction of snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # check danger straight/ahead
            (dir_r and game.is_collision(point_r)) or #if snake is going right, and point right of snake is_collision
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # check danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # check danger left            
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # convert list to np array (and change bools to ints)

    def remember(self, state, action, reward, next_state, done):
        """
        MEMORY
        """
        self.memory.append((state, action, reward, next_state, done)) # using deque append method
        # will popleft if MAX_MEMORY is reached.

    def train_long_memory(self):
        """
        train on batch of samples from memory
        """
        if len(self.memory) > BATCH_SIZE: # if we have over 1000 samples in memory...
            mini_sample = random.sample(self.memory, BATCH_SIZE) # get a random sample, list of tuples.
        else: # if we do not yet have 1000 samples in memory...
            mini_sample = self.memory
        
        # extract all states from all memory 'state', and do same for actions etc...
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        train on single step
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        random moves: tradeoff exploration / exploitation
        epsilon is hard coded equation, can play around with this:-
        -> more games we have, smaller epsilon gets
        -> smaller epsilon gets, the less randomness comes into play
        """
        # 
        self.epsilon = 80 - self.n_games #  
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # then take a random move
            move = random.randint(0,2)
            final_move[move] = 1
        else: # do a move based on model
            state0 = torch.tensor(state, dtype=torch.float) # convert state to tensor
            prediction = self.model(state0) # executes forward function, returns tensor x (nn outout)
            move = torch.argmax(prediction).item() #.item() to convert tensor to integer... i.e. [1,0,0] -> 0, [0,1,0] -> 1
            """
            e.g.    [5.0, 2.7, 0.1] (float output of NN)
            max ->
                    [1,0,0] 
            """
            final_move[move] = 1

        return final_move


def train():
    plot_scores = [] # keep track of scores
    plot_mean_scores = [] # keep track of mean scores
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True: # training loop
        # get old state (current state)
        state_old = agent.get_state(game)

        # get move (based on current state)
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #  remember (store in memories)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (replay memory/experience replay)
            # trains again on all previous moves in all previous games...
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()  # if we have new high score, we save model

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores) 

if __name__ == '__main__':
    train()
