import go
import nn
import params

class SNode():
    '''represents one possible state of the game
    stores:
    -a visit count 
    -a set of edges that correspond to all possible actions one can take
    -a parent action
    '''
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        #initialized to zero when created, will be updated later
        self.vists = 0
        self.actions = []
        #added once state is evaluated
        self.valuation = None

    def expand(self, network):
        pass

    def expanded(self):
        return self.visits!=0

class AEdge():
    '''represents a possible action in a state
    stores:
    -the move itself
    -the resulting state
    -the parent state
    -a prior chance of being played
    -total subtree action value
    -visit count
    -action value (in (-1,1) ) (equal to subtree action value over visits)
    '''
    def __init__(self, x, y, parent, prior):
        self.prior = prior
        self.parent = parent
        self.move = x,y
        self.visits = 0
        self.subtree_value = 0
        self.action_value = 0
        #calculate successor
        self.result = parent.copy()
        self.result.move(x,y)

    def __uct(self):
        return params.c_puct*self.prior*(self.parent.visits**.5)/(1+self.visits)

    def value(self):
        return self.action_value + self.__uct()
    
    def visit(self):
        self.visits += 1
        return self.result

    
def select(state):
    pass

def backup(state):
    pass

def search(state, network, playouts=100):
    #expand root
    for _ in range(playouts):
        #select next node
        #expand node
        #backup node
        pass

def pick_move(tree):
    pass
