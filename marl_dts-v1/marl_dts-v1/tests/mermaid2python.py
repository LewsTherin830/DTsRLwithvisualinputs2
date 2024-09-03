class Node:
    def __init__(self, condition):
        self.fm, self.sm = condition.split("<")
        self.fm = int(self.fm.replace("x_", ""))
        self.sm = float(self.sm)
        self.left = None
        self.right = None

    def get_output(self, state):
        if state[self.fm] < self.sm:
            return self.left.get_output(state)
        else:
            return self.right.get_output(state)


class Leaf(Node):
    def __init__(self, value):
        self.value = value

    def get_output(self, state):
        return self.value


def convert(string):
    root = None
    nodes = {}

    for l in string.split("\n"):
        if "-->" in l:
            from_, branch, to = l.split(" ")

            if "true" in branch:
                nodes[from_].left = nodes[to]
            else:
                nodes[from_].right = nodes[to]
        elif "<" in l:
            id_, cond = l.replace("]", "").split(" [")
            nodes[id_] = Node(cond)
            if len(nodes) == 1:
                root = nodes[id_]
        else:
            id_, value = l.replace("]", "").split(" [")
            nodes[id_] = Leaf(int(value))
            if len(nodes) == 1:
                root = nodes[id_]
    return root
