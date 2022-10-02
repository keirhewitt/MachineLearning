import numpy as np

#i_inputs = np.random.rand(5)   # Inputs = random array [between] 0 and 1
i_inputs = [0.2, 0.4, 0.1, 0.5, 0.8]
t_outputs = np.random.randint(5, size=5)   # Target outputs = array of 0s and 1s
#w_weights = np.random.rand(5)   # Weights = random array [between] 0 and 1
w_weights = [0.4, 0.2, 0.9, 0.3, 0.2]
lr = 0.1    # Learning rate
threshold = 0.8

class Perceptron:
    def __init__(self, inputs, t_outputs, weights, threshold, lr, training_loops=1000):
        self.training_loops = training_loops
        self.lr = lr
        self.inputs = inputs
        self.expected_outputs = t_outputs
        self.actual_outputs = []
        self.weights = weights
        self.bias = 1
        self.weighted_input = 0
        self.threshold = threshold
        
    def summation(self):
        sum = 0
        self.actual_outputs = []
        for idx, i in enumerate(self.inputs):
            sum += i * self.weights[idx]
            self.actual_outputs.append(sum)
        return sum

    def calculate_weighted_input(self):
        self.weighted_input = self.summation()
        return self.activation_function(self.weighted_input)

    # def assess_results(self):
    #     for i in self.expected_outputs:
    #         for j in self.actual_outputs:
    #             if i != j:
    #                 self.update_weights()
    #     print("Solution found!")
    #     return True              

    def activation_function(self, x):
        #print(f'Assessing output: {str(x)}')
        return 1 if x > self.threshold else 0

    def update_weights(self):
        #print("Updating weights...")
        updated_weights = []
        for idx, i in enumerate(self.weights):
            # wi <-- wi + Δwi
            # Δwi = lr(target - actual)xi
            new_weight = i + (self.lr * (self.expected_outputs[idx] - self.actual_outputs[idx]) * self.inputs[idx])
            updated_weights.append(new_weight)
        self.weights = updated_weights

    
loops_taken = 0
p = Perceptron(i_inputs, t_outputs, w_weights, threshold, lr)

print(f'Original weights: {str(p.weights)}')
print(f'Expected outputs: {str(t_outputs)}')
for _ in range(p.training_loops):
    loops_taken += 1
    if (p.calculate_weighted_input() == 1):
        break
    else:
        p.update_weights()
        #print(f'Output: {str(p.actual_outputs)}')

print(f'Updated weights: {str(p.weights)}')
print(f'Took {str(loops_taken)} loops to pass.')
