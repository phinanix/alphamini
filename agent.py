import mcts

class Agent():
    def __init__(self, network):
        self.network = network
        self.search_tree = None

    def move(self, state, temp, retain_tree=True, playouts=100):
        #initialize search tree
        if self.search_tree:
            assert self.search_tree.state == state
        else:
            self.search_tree = mcts.SNode(state, None)
        #search for specified playouts
        mcts.search(self.search_tree, self.network, playouts=playouts)
        #pick move
        picked_move = mcts.pick_move(self.search_tree, temp)
        if retain_tree:
            for child in self.search_tree.actions:
                if child.move==picked_move:
                    self.search_tree = child
            assert self.search_tree.move = picked_move, \
                            "must have properly replaced the search tree"
            self.search_tree = self.search_tree.child
        else:
            self.search_tree = None
        return picked_move
                    