import torch
import torch.nn as nn

from tqdm import tqdm

from utils import *

def main():

    n_hidden = 128
    model = RNN(n_letters, n_hidden, n_categories)

    lr: float = 0.005
    n_epochs: int = 100000
    log_every: int = 500
    logger = Logger()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    current_loss = 0

    def train(category_tensor, line_tensor):
        model.zero_grad()
        hidden = model.init_hidden()
        
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        optimizer.step()

        return output, loss.item()
    
    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        # Get a random training input and target
        category, line, category_tensor, line_tensor = random_training_pair()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        
        if epoch % log_every == 0:
            pbar.set_description(f'Loss: {current_loss / log_every:.4f}')
            logger.log(current_loss / log_every)
            current_loss = 0

    # Keep track of correct guesses in a confusion matrix
    accuracy, confusion_matrix = get_accuracy_and_confusion_matrix(model)
    logger.add_figure('Fig1', confusion_matrix)
    logger.add_accuracy(accuracy)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class NameClassifierGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameClassifierGRU, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fc(output[-1])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)




if __name__ == '__main__':
    main()