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
        self.board = np.zeros((size, size, hist_size*2))
        self.cur_player = 0 #black first
        self.komi = komi
        self.turn = 0


    '''
    '''
    def __in_bounds(self, i, j):
        return i >= 0 and j >= 0 and i < self.size and j < self.size
    '''returns the orthogonal neighbors of a given point
    '''
    def __neighbors(self, i, j):
        candidates = [(i,j-1), (i,j+1), (i-1,j), (i+1,j)]
        out = []
        for x,y in candidates:
            if self.__in_bounds(x,y):
                out.append( (x,y) )
        return out
    
    '''takes a board, an intersection, and a color
    returns a list of tuples of intersections containing all squares that 
    stone is connected to
    '''
    def __group(self, board, i, j, color):
        if board[i,j,color]==0:
            return []
        out = [(i,j)]
        s = self.__neighbors(i,j)
        visited = set()
        #depth first search to find the whole group
        while s:
            x,y = s.pop()
            visited.add( (x,y) )
            if board[x,y,color] == 1:
                out.append( (x,y) )
                s.extend([x for x in self.__neighbors(x,y) if x not in visited])
        return out
        
    '''takes a sxsx2 array an intersection and a color
    counts the liberties of the group which the stone at i,j is part of
    returns -1 if the space is empty'''
    def __liberties(self, board, i, j, color, verbose=False):
        if board[i,j,color]==0:
            if verbose:
                print("recieved board", self.__print_board(board))
            return -1
        liberty_set = set()
        for x,y in self.__group(board, i, j, color):
            liberty_set.update(self.__neighbors(x,y))

        liberty_count = 0
        for x,y in liberty_set:
            if board[x,y,0]==0 and board[x,y,1]==0:
                liberty_count += 1
        return liberty_count
    
    '''takes a size x size x 2 array representing a board and a color
    returns the board with all stones of that color captured
    '''
    def __capture(self, board, color, verbose=False):
        new_board = np.copy(board)
        for i in range(self.size):
            for j in range(self.size):
                if self.__liberties(new_board,i,j, color) == 0:
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
        
    '''plays a move for the current player at (x,y)
    prints an error message if False is returned, no message if True
    returns True if sucessful, False if unsuccessful
    error parameter controls verbosity
    '''
    def move(self, x, y, error=False):
        #check legality
        if not self.is_legal(self.board,x,y,self.cur_player,error=error):
            return False
        #update history
        self.board[:,:,2:] = self.board[:,:,:-2]
        #last 2 boards are now duplicated
        #modify board
        #print(self.board)
        self.board[x,y,self.cur_player] = 1
        #print(self.board)
        #clear captures
        self.board[:,:,:2] = self.__capture(self.board[:,:,:2],
                                            (self.cur_player+1)%2)
        self.board[:,:,:2] = self.__capture(self.board[:,:,:2],
                                            self.cur_player)
        #update turn and player
        self.cur_player = (self.cur_player + 1) % 2
        self.turn += 1
        #success!
        return True

    def i_move(self, x, y):
        if self.move(x,y,error=True):
            print('New Board:')
            print(self.get_board_str)

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

    def __print_board(self, board):
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
           return self.__print_board(self.board)
    def get_history_str(self):
        out = []
        for i in range(0, self.hist_size*2, 2):
            out.append(self.__print_board(self.board[:,:,i:i+2]))
        return '\n'.join(out)
