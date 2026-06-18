from nanograd.engine.tensor import Tensor
from graphviz import Digraph


def trace(root):
    """
    root is a Tensor.
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
    WORKS ONLY WITH SCALARS ATM
    Generates a Graphviz visualization of the computation graph.
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


