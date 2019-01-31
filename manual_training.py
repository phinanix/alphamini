from training_test import *
import cProfile
import timeit
test = TestTraining()

start_time = timeit.default_timer()
#cProfile.run("test.test_train_loop()", "profiling_results.txt")
test.test_parallel_loop()
print("time:", timeit.default_timer() - start_time)
