import math
from statistics import mean
import numpy as np

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
        self.visits = 0
        self.actions = []
        #added once state is evaluated        #added once state is evaluated

        self.valuation = None

    '''the policy output of the network is treated as 
    the direct probability, clipped between 0 and 1
    '''
    def expand(self, network):
        assert not self.is_expanded(), "expanded an already expanded node"
        self.visits = 1
        #feed my state to the network
        policy, value = network.evaluate(self.state.board,
                                         self.state.cur_player)
        policy = policy.flatten()
        #first element of policy array is pass
        #label self valuation
        self.valuation=value[0][0]
        #generate children
        legal_moves = self.state.legal_moves(self.state.cur_player)
        #passing is always legal
        pass_child = AEdge(-1, -1, self, math.e**policy[0])
        self.actions.append(pass_child)
        #reshape priors
        priors = policy[1:].reshape(self.state.size, self.state.size)
        for x in range(self.state.size):
            for y in range(self.state.size):
                if legal_moves[x,y]:
                    #label children with priors
                    prior = priors[x,y]
                    #print("prior:", prior)
                    child = AEdge(x,y,self, prior)
                    self.actions.append(child)
        return self.valuation
        
    def is_expanded(self):
        return self.visits!=0

    def best_child(self):
        if not self.actions:
            return None
        best = self.actions[0]
        for child in self.actions[1:]:
            if child.value() > best.value():
                best = child
        return best

    def visit(self):
        self.visits += 1
        return self.best_child()

    def value(self):
        if self.actions:
            return mean( x.action_value for x in self.actions )
        else:
            return 0

    def policy(self, temp=1):
        move_array = np.zeros( (self.state.size,self.state.size) )
        for action in self.actions:
            x,y = action.move
            if x==-1 and y==-1:
                pass_count = action.visits
            else:
                move_array[x,y] = action.visits
        if temp == 0:
            max_val = max(pass_count, move_array.max())
            temps = np.zeros_like(move_array)
            pass_temp = 0
            if pass_count >= max_val:
                pass_temp = 1
            temps[np.where(move_array == max_val)] = 1
        else:
            temps = np.power(move_array, 1/temp)
            pass_temp = pass_count**(1/temp)
        probs = np.divide(temps, temps.sum()+pass_temp)
        pass_prob = pass_temp / (temps.sum()+pass_temp)
        one_d_probs = np.zeros( probs.size+1 )
        one_d_probs[1:] = np.ravel(probs)
        one_d_probs[0] = pass_prob
        return one_d_probs

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
        result = parent.state.copy()
        result.move(x,y)
        self.child = SNode(result, self)
        
    def __uct(self):
        return params.c_puct*self.prior*(self.parent.visits**.5)/(1+self.visits)

    def value(self):
        return self.action_value + self.__uct()
    
    def visit(self):
        self.visits += 1
        return self.child

    def update(self, child_value):
        #TODO: figure out why sometimes a node is being updated but has
        #been visited 0 times
        self.subtree_value += child_value
        #print("updated action value to:",
        #self.subtree_value / max(self.visits, 1))
        self.action_value = self.subtree_value / max(self.visits, 1)

'''traverses the tree starting at state to determine next node
to expand
current is the current state in the tree
'''
def select(state):
    current = state 
    while current.is_expanded():
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
    #expand root#I don't know what this comment means but I think the root is
    #taken care of?
    
'''
runs a specified number of playouts 
starting from a given root and network
returns the evaluated root
'''
def search(root, network, playouts=100):
    root.expand(network)
    for i in range(playouts):
        leaf = select(root)
        leaf.expand(network)
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
    one_d_probs = root.policy(temp=temp)
    #print("one_d_probs:", one_d_probs)
    choice = np.random.choice(np.arange(size**2+1), p=one_d_probs)
    if choice == 0:
        return (-1,-1) #decided to pass
    #-1 since passing occupies 0, but should not be counted
    return np.unravel_index(choice-1, (size,size))
    
