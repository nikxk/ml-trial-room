from io import open
import glob
import os

import unicodedata
import string
import random
random.seed(0)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    # Read a file and split into lines
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor
    
category_lines = {}
all_categories = []

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def category_from_output(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

class Logger:
    def __init__(self):
        self.writer = SummaryWriter()
        self.step = 0
    def log(self, loss):
        self.writer.add_scalar('loss', loss, self.step)
        self.step += 1
    def add_figure(self, name, figure):
        self.writer.add_figure(name, figure)
    def add_accuracy(self, accuracy):
        self.writer.add_scalar('accuracy', accuracy, self.step)

def get_accuracy_and_confusion_matrix(model: nn.Module):
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    correct = 0

    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = model.init_hidden()
        
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)
        
        return output

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_pair()
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
        if guess == category:
            correct += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return correct/n_confusion, plt.gcf()
