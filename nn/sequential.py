from nn.modules import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.layers=layers
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def forward(self, x):
        for layer in self.layers:
            x=layer.forward(x)
        return x