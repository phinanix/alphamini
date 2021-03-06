import math
import numpy as np

'''
design decisions:
tromp-taylor rules
suicide is forbidden
simple ko
'''

class GoGame():
    '''internal representation of board:
    size x size x 2 array is a board, 1 occupied, 0 not
    first array is black, second white
    the object keeps track of history for hist_size steps
    rules of the game require a minimum of 3 to track ko
    black is player 0, white player 1
    '''
    def __init__(self, size, hist_size, komi):
        self.hist_size = max(hist_size, 3) #for ko
        self.size = size
        self.komi = komi
        self.board = np.zeros((size, size, hist_size*2))
        self.cur_player = 0 #black first
        self.turn = 0
        self.pass_count = 0
        self.game_over = False
        self.turn_limit = math.floor(1.5* self.size**2)

    '''returns a deep/true copy of the class
    '''
    def copy(self):
        out = GoGame(self.size, self.hist_size, self.komi)
        out.board = np.copy(self.board)
        out.cur_player = self.cur_player
        out.turn = self.turn
        return out

    def cur_board(self):
        return self.board[:,:,:2]
    
    '''returns whether a board position is in bounds of the board
    '''
    def __in_bounds(self, i, j):
        return i >= 0 and j >= 0 and i < self.size and j < self.size

    '''returns the orthogonal neighbors of a given point
    '''
    def __neighbors(self, i, j):
        candidates = [(i,j-1), (i,j+1), (i-1,j), (i+1,j)]
        return [(x,y) for x,y in candidates if self.__in_bounds(x,y)]
        '''out = []
        for x,y in candidates:
            if self.__in_bounds(x,y):
                out.append( (x,y) )
        return out'''

    def __neighbors_of_group(self, group_list):
        group_set = set(group_list)
        candidates = set()
        for x,y in group_set:
            candidates.update(self.__neighbors(x,y))
        return candidates - group_set
        
    '''takes a board, an intersection, and a color
    returns true if there is a stone of that color at that position
    or if color is -1, whether that intersection is empty 
    '''
    def __check_square(self, board, i, j, color):
        if color == -1:
            #check if square is empty
            return board[i,j,0]==0 and board[i,j,1]==0
        else:
            #check if square contains the given color
            return board[i,j,color] == 1

    def __check_squares(self, board, square_list, color):
        return any( (self.__check_square(board, i, j, color)
                     for i,j in square_list) )

    def __is_empty(self, board, i, j):
            return board[i,j,0]==0 and board[i,j,1]==0
    
    '''takes a board, an intersection, and a color
    returns a list of tuples of intersections containing all squares that 
    stone is connected to
    if color = -1, does the same, but for empty squares
    '''
    def __group(self, board, i, j, color):
        if not self.__check_square(board, i, j, color):
            return []
        out = [(i,j)]
        s = self.__neighbors(i,j)
        visited = set()
        #depth first search to find the whole group
        while s:
            x,y = s.pop()
            visited.add( (x,y) )
            if self.__check_square(board, x, y, color):
                out.append( (x,y) )
                s.extend([x for x in self.__neighbors(x,y) if x not in visited])
        return out
        
    '''takes a sxsx2 array an intersection and a color
    counts the liberties of the group which the stone at i,j is part of
    returns -1 if the space is empty'''
    def __liberties(self, board, i, j, color):
        if board[i,j,color]==0:
            #modified from -1 to true
            return True
        neighbors = self.__neighbors_of_group(self.__group(board,i,j,color))
        return any( self.__is_empty(board,x,y) for x,y in neighbors )

    def can_reach(self, board, i, j, color, color_to_reach):
        neighbors = self.__neighbors_of_group(self.__group(board,i,j,color))
        return any(self.__check_square(board,i,j,color_to_reach)
                   for i,j in neighbors)
    
    '''takes a size x size x 2 array representing a board and a color
    returns the board with all stones of that color captured
    '''
    def __capture(self, board, color, verbose=False):
        new_board = np.copy(board)
        for i in range(self.size):
            for j in range(self.size):
                if self.__liberties(new_board,i,j,color) == False:
                    if verbose:
                        print("capturing group at:", i, j, "color:", color)
                    for point in self.__group(new_board, i, j, color):
                        x = point[0]
                        y = point[1]
                        new_board[x,y,color] = 0
        return new_board

    '''internal function that puts a piece on the board and captures for the
    other color, then the given color'''
    def __update(self, board, x, y, color):
        out = np.copy(board)
        out[x,y,color] = 1
        out = self.__capture(out, (color+1)%2)
        return self.__capture(out, color)
    
    '''takes a board and a move 
    returns whether playing that move on that board is self-capture
    move is guaranteed to be an empty space'''
    def __self_capture(self, board, x, y, color):
        board_cp = np.copy(board)
        board_cp[x,y,color] = 1
        board_cp = self.__capture(board_cp, (color+1)%2)
        final_board = self.__capture(board_cp, color)
        return not np.array_equal(final_board, board_cp)
        #if they are not equal, then there was self-caputre

    '''takes a board, a past board and a move 
    returns whether that move is ko, ie whether it recreates the past board
    '''
    def __is_ko(self, board, old_board, x, y, color, verbose=False):
        if verbose:
            print("board:\n", self.__print_board(board))
            print("old_board:\n", self.__print_board(old_board))
        final_board = self.__update(board, x, y, color)
        return np.array_equal(final_board, old_board)
        #if they're equal, it's an illegal ko move

    def is_legal(self, board, x, y, color, error=False):
        #check validity
        if not self.__in_bounds(x,y):
            #move is outside the bounds
            if error:
                print('move is outside legal bounds')
            return False
        #check occupied
        if self.board[x,y,0] == 1 or self.board[x,y,1]==1:
            if error:
                print('move is on a stone already played')
            return False
        #check suicide
        if self.__self_capture(self.board[:,:,:2], x,y, color):
            if error:
                print('move is suicidal')
            return False
        #check ko
        if self.__is_ko(self.board[:,:,:2],
                        self.board[:,:,2:4],
                        x,y, color,
                        verbose=False):
            if error:
                print('move is ko')
            return False
        return True
    

    def is_over(self):
        return self.game_over
    
    '''plays a move for the current player at (x,y)
    prints an error message if False is returned, no message if True
    returns True if sucessful, False if unsuccessful
    error parameter controls verbosity

    (-1, -1) is a pass, which is always legal and does not change the board state
    '''    
    def move(self, x,y, error=False, score_callback=None):
        if self.is_over():
            if error:
                print('game is over')
            return False
        #check legality
        if not self.is_legal(self.board,x,y,self.cur_player,error=error):
            return False
        return self.unsafe_move(x,y,error=error,score_callback=score_callback)
    
    '''makes a move for the current player at (x,y)
    does not check legality of move - this could cause bugs !!
    '''
    def unsafe_move(self, x, y, error=False, score_callback=None):
        
        if x == -1 and y == -1:
            
            #this is a pass
            self.cur_player = (self.cur_player + 1) % 2
            self.turn += 1
            self.pass_count += 1
            if self.pass_count == 2:
                if error:
                    print('game ended')
                #game ends
                self.game_over=True
                if score_callback:
                    score_callback(self.score())
            if self.turn > self.turn_limit:
                if error:
                    print('game ended on turns')
                self.game_over=True
                if score_callback:
                    score_callback(self.score())
            #success!
            return True

        #update history
        self.board[:,:,2:] = self.board[:,:,:-2]
        #last 2 boards are now duplicated
        #modify board
        #print(self.board)
        self.board[x,y,self.cur_player] = 1
        #print(self.board)
        #clear captures
        self.board[:,:,:2] = self.__capture(self.board[:,:,:2],
                                            (self.cur_player+1)%2,
                                            verbose = False)
        self.board[:,:,:2] = self.__capture(self.board[:,:,:2],
                                            self.cur_player,
                                            verbose = False)
        #update turn and player
        self.cur_player = (self.cur_player + 1) % 2
        self.turn += 1
    
        if self.turn > self.turn_limit:
            if error:
                print('game ended on turns')
            self.game_over=True
            if score_callback:
                score_callback(self.score())
        #success!
        return True

    def i_move(self, x, y):
        if self.move(x,y,error=True):
            print('New Board:')
            print(self.get_board_str())

    def play_moves(self, move_list, error=False):
        for x,y in move_list:
            self.move(x,y, error=error)

    '''returns an array with 1s at legal moves and 0s at illegal moves'''
    def legal_moves(self, color):
        out = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if self.is_legal(self.board[:,:,:2],i,j,color):
                    out[i,j] = 1
        return out

    def __one_score(self, board, color):
        pass
        #scan over board, marking 
        #

    def __mark(self, board, l, value):
        for x,y in l:
            board[x,y] = value
            
    '''returns the current score if the game were to immediately end
    from the perspective of a given color
    '''
    def score(self, color, verbose = False):
        other_color = (color+1)%2
        if color == 1:
            adj_komi = self.komi
        else:
            adj_komi = -1 * self.komi
        
        final_board = self.board[:,:,:2]
        #this array will contain 1s at one's stones and -1s at unfriendly stones
        marking = self.board[:,:,color] - self.board[:,:,other_color]
        #fill in empty squares on the board with the person they belong to
        #or NaN if they do not belong to people
        for x in range(self.size):
            for y in range(self.size):
                if marking[x,y] == 0:
                    group = self.__group(final_board,x,y,-1)
                    reach_friend = self.can_reach(final_board,x,y,-1,color)
                    reach_foe = self.can_reach(final_board,x,y,-1,other_color)
                    if reach_friend and reach_foe:
                        self.__mark(marking, group, np.nan)
                    elif reach_friend:
                        self.__mark(marking, group, 1)
                    elif reach_foe:
                        self.__mark(marking, group, -1)
                    else:
                        #this means there are no stones on the board
                        assert not np.any(final_board)
                        self.__mark(marking, group, 0)
        if verbose:
            print("marked board:\n", marking)
            print("komi:", adj_komi)
        return np.nansum(marking) + adj_komi

    '''returns the game result, 1 for a black win, -1 for a white win
    '''
    def result(self):
        if self.score(0) > 0:
            return 1
        else:
            return -1
        
    def print_board(self, board):
        out = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i,j,0] == 1:
                    out.append('B')
                elif board[i,j,1] == 1:
                    out.append('W')
                else:
                    out.append('.')
            out.append('\n')

        return ''.join(out)
        
    def get_board_str(self):
           return self.print_board(self.board)
    def get_history_str(self):
        out = []
        for i in range(0, self.hist_size*2, 2):
            out.append(self.__print_board(self.board[:,:,i:i+2]))
        return '\n'.join(out)
