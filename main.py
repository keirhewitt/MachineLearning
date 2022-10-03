import numpy as np
from perceptron import Perceptron

i_X = [0.2, 0.4, 0.1, 0.5, 0.8]
Y = np.random.randint(5, size=5)   # Target outputs = array of 0s and 1s
w_w = [0.4, 0.2, 0.5, 0.3, 0.2]
lr = 0.1
threshold = 0

"""Format each number in array to 2 decimal places"""
def array_formatter(arr):
    formatted = ["{:0.2f}".format(x) for x in arr ]
    return formatted

"""Function for training a perceptron object {perceptron, number of loops to train it}"""
def train_model(perceptron, loops):
    loops_taken = 0
    for _ in range(loops):
        loops_taken += 1
        if (loops_taken == 2): 
            print(f'1st Outputs: {str(array_formatter(perceptron.y))}')
        if (perceptron.calculate_weighted_input() == 1):
            break
        else:
            perceptron.update_w()

    print(f'\nExpected outputs: {str(Y)}')
    print(f'Final Outputs: {str(array_formatter(perceptron.y))}')
    print(f'Took {str(loops_taken)} loops to pass.\n')

p = Perceptron(i_X, Y, w_w, threshold, lr)

train_model(p, 10000)