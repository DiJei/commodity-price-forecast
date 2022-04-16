import time
import torch

from torch import nn
from torch import optim



class Trainer(object):

    def __init__(self, model, config) -> None:
        super().__init__()
        self.model = model
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.data_loader = config['data_loader']
        self.optim = config['optimizer']
        self.criterion = config['criterion']

    def run_epoch(self):
        """Runs an epoch of training.
        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.
        Returns:
        - The epoch loss (float).
        """

        epoch_loss = 0.0
        losses = []
        for step, batch_data in enumerate(self.data_loader):
            
            # Forward propagation
            outputs = self.model(batch_data['x'])

            # Loss computation
            loss = self.criterion(outputs, batch_data['y'])

            # Backpropagation
            self.optim.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            losses.append(loss.item())

        return epoch_loss / len(self.data_loader)

class Tester(object):

    def __init__(self, model, config) -> None:
        super().__init__()
        self.model = model
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.data_loader = config['data_loader']
        self.optim = config['optimizer']
        self.criterion = config['criterion']

    def run_epoch(self):

        epoch_loss = 0.0
        for step, batch_data in enumerate(self.data_loader):

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(batch_data['x'])

                # Loss computation
                loss = self.criterion(outputs, batch_data['y'])

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

        return epoch_loss / len(self.data_loader)