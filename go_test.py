import go

test_moves = [(1,1), (1,2), (2,2), (0,1), (0,2)]
game = go.GoGame(3, 3, 4.5)
game.play_moves(test_moves)
print(game.get_board_str())
game.i_move(1,2)
print(game.legal_moves(0))
print(game.legal_moves(1))
