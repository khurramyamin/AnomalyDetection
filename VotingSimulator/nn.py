# Simple neural network to make predictions. Change this later on?

from numpy import argmax
import torch

class TinyModel(torch.nn.Module):

    def __init__(self, input_size, output_size=1, dropout_included=False, dropout_prob=0):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, input_size*2)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_size*2, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.dropout_included = dropout_included

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        if (self.dropout_included):
            x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def predict(self, list):
        return self.forward(torch.FloatTensor(list)).item()
        #return torch.argmax(self.forward(torch.FloatTensor(list))).item()

# tinymodel = TinyModel(3, 2)

# print('The model:')
# print(tinymodel)

# print('\n\nJust one layer:')
# print(tinymodel.linear2)

# print('\n\nModel params:')
# for param in tinymodel.parameters():
#     print(param)

# print('\n\nLayer params:')
# for param in tinymodel.linear2.parameters():
#     print(param)

# print('\n\nWeights:')
# print(tinymodel.linear1.weight.data)
# print(tinymodel.linear2.weight.data)

# a = tinymodel.forward(torch.FloatTensor([3, 44, 2.5]))
# print(a)
# print(torch.argmax(a))