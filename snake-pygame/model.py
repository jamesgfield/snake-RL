import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        forward function (pass tensor x through network)
        """
        # apply linear layer:
        x = F.relu(self.linear1(x)) # put tensor through linear layer, then apply activation function
        x = self.linear2(x) # no activation used, raw numbers returned as outputs
        return x

    def save(self, file_name='model.pth'):
        """
        helper function to save the model
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # optimizer:
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # loss-function (MSE):
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        train nn (either on one, or multiple samples)
        """
        # convert inputs to pytorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # handle multiple sizes:
        # (n, x)

        if len(state.shape) == 1:
            # want in the form: (1, x) where 1 is number of batches
            # can append 1 dimension with torch.unsqueeze function
            state = torch.unsqueeze(state, 0) # want to append a dimension in 0th position
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # also want to convert done value from single value to tuple
        # Bellman Equation implementation

        # 1: want predicted Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + (gamma) * max(next_predicted Q value) -> only do if not done
        # max(next_predicted Q value) -> this gives only one value
        # action is 3 different values, so must clone:
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad() # empty gradient (pytorch)
        loss = self.criterion(target, pred)
        loss.backward() # apply back propagation

        self.optimizer.step()





