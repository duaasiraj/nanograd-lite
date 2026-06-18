class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad=np.zeros_like(p.data)
    
    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)