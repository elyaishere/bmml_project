import torchvision.models as models
import torch.nn as nn

class Resnet(nn.Module):
  def __init__(self, name, num_classes):

    assert name in ['resnet50', 'resnet101', 'resnet152']
    super(Resnet, self).__init__()

    if name == 'resnet50':
      self.model = models.resnet50()
    elif name == 'resnet101':
      self.model = models.resnet101()
    else:
      self.model = models.resnet152()
    
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    self._model_name = name
  
  def forward(self, input):
    return self.model(input)
