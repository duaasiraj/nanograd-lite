from engine.tensor import Tensor
from viz.graph_viz import draw_dot , trace 
from graphviz import Digraph

x = Tensor(2.0)
y = Tensor(3.0)
z = x * y + x
z.backward()
print(x.grad) 
print(y.grad) 

dot = draw_dot(z)
dot.render("graph", view=True)  