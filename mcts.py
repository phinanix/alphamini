import go
import nn
import params

class SNode():
    '''represents one possible state of the game
    stores:
    -a visit count 
    -a set of edges that correspond to all possible actions one can take
    -a parent action (root's parent is None)
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
        self.visits = 1
        #feed my state to the network
        #label self valuation
        #generate children
        #label children with priors
        return self.valuation
        
    def is_expanded(self):
        return self.visits!=0

    def best_child(self):
        best = self.actions[0]
        for child in self.actions[1:]:
            if child.value() > best.value:
                best = child
        return best

    def vist(self):
        self.visits += 1
        return self.best_child()

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

    def update(self, child_value):
        self.subtree_value += child_value
        self.action_value = self.subtree_value / self.visits

'''traverses the tree starting at state to determine next node
to expand
current is the current state in the tree
'''
def select(state):
    current = state 
    while not current.is_expanded():
        action = current.visit()
        current = action.visit()
    return current

'''backs up the tree, starting at state, which is a node just evaluated
to update the tree above state with it's newly calculated value
current is the current action we're updating
'''
def backup(state):
    value = state.valuation
    current = state.parent
    current.update(value)
    while current.parent.parent:
        current = current.parent.parent
        current.update(value)
    #expand root

'''
runs a specified number of playouts 
starting from a given root and network
returns the evaluated root
'''
def search(root, network, playouts=100):
    for _ in range(playouts):
        leaf = select(root)
        leaf.expand()
        backup(leaf)
    return root

'''
picks a move with some randomness given the root of a search tree
temperature determines randomness, temp=0 picks the best move, while
progressivly higher temperatures increas the probability of picking worse
moves
'''
def pick_move(root, temp):
    size = root.state.size
    move_array = np.zeros( (size,size) )
    for action in root.actions:
        x,y = action.move
        move_array[x,y] = action.visits
    if temp != 0:
        temps = np.power(move_array, 1/temp)
    else:
        temps = np.zeros_like(move_array)
        temps[np.where(move_array == move_array.max())] = 1
    probs = np.divide(temps, temps.sum())
    1d_probs = np.ravel(probs)
    choice = np.random.choice(np.arrange(size**2), p=1d_probs)
    return np.unravel_index(choice, temps.shape)
    
