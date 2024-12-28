import torch
from torch import nn
from torch.nn import functional as F

def reset_weights(m):
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			layer.reset_parameters()

def binaryClassifierHiddenUnitIterator():
	for i in range(10):
		yield i + 1

class BinaryClassifier(nn.Module):
   # Layer {1,2,3,4,5,6,7,8,9,10}
  def __init__(self,input_shape, hidden_layer=1):
    super(BinaryClassifier,self).__init__()
    self.fc1 = nn.Linear(input_shape,hidden_layer)
    self.fc2 = nn.Linear(hidden_layer,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    return x

def multiClassifierHiddenUnitIterator():
	for layer_1_hidden_unit in [15,25,50]:
		for layer_2_hidden_unit in [5,10,20]:
			yield layer_1_hidden_unit,layer_2_hidden_unit

class MultiClassClassifier(nn.Module):
  # Layer_1 {15, 25, 50}, Layer_2 {5, 10, 20}
  def __init__(self,input_shape, hidden_layer_1=15, hidden_layer_2=5):
    super(MultiClassClassifier,self).__init__()
    self.fc1 = nn.Linear(input_shape,hidden_layer_1)
    self.fc2 = nn.Linear(hidden_layer_1,hidden_layer_2)
    self.fc3 = nn.Linear(hidden_layer_2,10)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x