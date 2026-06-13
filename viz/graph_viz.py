from engine.tensor import Tensor
from graphviz import Digraph


# $env:PATH += ";C:\Program Files (x86)\Graphviz\bin" (fix this permenantly later)
# python -m viz.graph_viz (this too)

def trace(root):
    """
    Recursively traverses the computation graph starting from the output
    tensor and collects all nodes and edges required for visualization.

    Parameters
    ----------
    root : Tensor
        Final output tensor of the computation graph.

    Returns
    -------
    tuple[set, set]
        nodes : All tensors participating in the graph.
        edges : Parent -> child relationships between tensors.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """
    Generates a Graphviz visualization of the computation graph.

    Each tensor node displays its current data value and gradient.
    Operation nodes (+, *, etc.) are rendered separately and connected
    to the tensors they produce, making the forward and backward
    pass structure explicit.

    Parameters
    ----------
    root : Tensor
        Final output tensor of the graph.

    Returns
    -------
    graphviz.Digraph
        Renderable graph object. Call .render() or .view() to export.
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(
            name=uid,
            label="{ data %.4f | grad %.4f }" % (n.data, n.grad),
            shape="record",
        )

        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


a = Tensor(2.0)
b = Tensor(3.0)
c = a + b
d = c * b

dot = draw_dot(d)
dot.render("graph", view=True)  