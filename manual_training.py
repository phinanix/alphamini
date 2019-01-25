from training_test import *
import cProfile
test = TestTraining()
cProfile.run("test.test_train_loop()", "profiling_results.txt")
