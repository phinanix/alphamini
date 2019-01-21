import go

from hypothesis import given, assume
from hypothesis.strategies import floats, integers, tuples, lists, composite

size = 9
moves = lists(tuples(integers(min_value=0, max_value = size-1),
                         integers(min_value=0, max_value = size-1)),
                  min_size = 1, max_size = 10, unique = True)

class TestGoClass():
    def set_up(self):
        game = go.GoGame(3,3,4.5)
        test_moves = [(1,1), (1,2), (2,2), (0,1), (0,2), (1,0), (2,0)]
        for x,y in test_moves:
            game.move(x,y, error=True)
        return game
    
    @given(integers(min_value=1, max_value = 100),
           integers(min_value=1, max_value = 100),
           floats(allow_nan=False) )
    def test_generation(self, size, hist_length, komi):
        game = go.GoGame(size, hist_length, komi)

    @given(moves)
    def test_random_moves(self, move_list):
        game = go.GoGame(size, 5, 4.5)
        for x,y in move_list:
            if game.is_legal(game.board[:2],x,y,game.cur_player):
                assert game.move(x,y)
        assert game.score(0) > -100
        assert game.score(0) < 100
        assert game.score(1) > -100
        assert game.score(1) < 100
        print(game.get_board_str())

    def test_specific_moves(self):
        game = go.GoGame(3,3,4.5)
        test_moves = [(1,1), (1,2), (2,2), (0,1), (0,2), (1,0), (2,0)]
        for x,y in test_moves:
            assert game.move(x,y, error=True)
        assert game.score(0)==-1.5
        assert game.score(1)==1.5

    @given(moves)
    def test_history(self,moves):
        pass
    
    @given(moves, moves)
    def test_legality(self,moves, locations):
        game = go.GoGame(size, 5, 4.5)
        game.play_moves(moves)
        
        legality = game.legal_moves(game.cur_player)
        for x,y in locations:
            test_game = game.copy()
            if legality[x,y]:
                assert test_game.move(x,y)
            else:
                assert not test_game.move(x,y)
    def test_suicide(self):
        game = self.set_up()
        assert game.move(0,0)==False
        game.cur_player=0
        game.move(2,1)
        game.cur_player=0
        assert game.move(1,2)==False

    def test_ko(self):
        game = go.GoGame(3,3,4.5)
        test_moves = [(1,1), (1,2), (2,2), (0,1), (0,2)]
        for x,y in test_moves:
            game.move(x,y, error=False)        
        assert game.move(1,2)==False
