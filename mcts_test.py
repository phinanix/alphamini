import go
import mcts
import nn

class TestMCTS():
    def gen_root(self):
        game = go.GoGame(9,8,4.5)
        t = mcts.SNode(game, None)
        return t
    def gen_tree(self):
        t = self.gen_root()
        child1 = mcts.AEdge(1,2,t,0.7)
        child2 = mcts.AEdge(0,0,t,0.3)
        t.actions = [child1, child2]
        return t,child1
    def gen_network(self):
        return nn.Network(9,8,
                          128,10,
                          32,32,256)
    def test_expand(self):
        network = self.gen_network()
        
        tree = self.gen_tree()[0]
        assert tree.is_expanded() == False
        tree.expand(network) #network is not implemented yet
        assert tree.is_expanded() == True

    def test_visits(self):
        tree=self.gen_tree()[0]
        assert tree.visits==0
        tree.visit()
        assert tree.visits==1
        tree.visit()
        assert tree.visits==2

    def test_best_child(self):
        tree,best_child=self.gen_tree()
        assert tree.best_child()==best_child

    def test_search(self):
        t = self.gen_root()
        network = self.gen_network()
        out = mcts.search(t, network, playouts=10)
        move = mcts.pick_move(out, .2)
