import go
import mcts

class TestMCTS():
    def gen_tree(self):
        game = go.GoGame(3,3,4.5)
        t = mcts.SNode(game, None)
        child1 = mcts.AEdge(1,2,t,0.7)
        child2 = mcts.AEdge(0,0,t,0.3)
        t.actions = [child1, child2]
        return t,child1
    
    def test_expand(self):
        tree = self.gen_tree()[0]
        assert tree.is_expanded() == False
        tree.expand(None) #network is not implemented yet
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
