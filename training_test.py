import training
import params as p
class TestTraining():
    def ze_setup(self, board_size):
        return training.Training(board_size)
    
    def test_init(self):
        train = self.ze_setup(9)
        
    def test_train_loop(self):
        train = self.ze_setup(p.board_size)
        train.training_loop('test_exp_rp','test_network',"train_log_file",
                            rounds=2, games_per_round=3, positions_per_round=128)

    def test_loading(self):
        train = training.Training(p.board_size,
                                  network_filename="test_load_network",
                                  exp_rp_filename="test_load_exp_rp.npz")
        train.training_loop('ER_load_test', 'network_load_test',
                            "train_log_file",
                            rounds=3, games_per_round=5,
                            positions_per_round=128)
