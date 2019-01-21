import nn
import go


class TestNN():
    def test_init(self):
        network = nn.Network(9,8,
                             128,10,
                             32,32,256)
    def test_predict(self):
        network = nn.Network(9,8,
                             128,10,
                             32,32,256)
        game = go.GoGame(9,8, 4.5)
        policy, value = network.evaluate(game.board, game.cur_player)
        print("policy:\n", policy, "value:", value)
        
tests = TestNN()
tests.test_predict()
