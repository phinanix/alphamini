import training

class TestTraining():
    def ze_setup(self, board_size):
        return training.Training(board_size)
    
    def test_init(self):
        train = self.ze_setup(9)
        
    def test_self_play(self):
        train = self.ze_setup(9)
        train.self_play(2, .3)
        train.self_train(num_positions=32)

