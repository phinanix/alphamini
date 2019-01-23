import training

class TestTraining():
    def ze_setup(self, board_size):
        return training.Training(board_size)
    
    def test_init(self):
        train = self.ze_setup(9)
        
    def test_train_loop(self):
        train = self.ze_setup(5)
        train.training_loop('test_exp_rp','test_network', rounds=3,
                            games_per_round=10, positions_per_round=64)

