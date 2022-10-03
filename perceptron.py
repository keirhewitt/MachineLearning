
class Perceptron:
    def __init__(self, X, Y, w, threshold, lr, training_loops=10000):
        self.training_loops = training_loops
        self.lr = lr    # Learning rate
        self.X = X  # Set of all inputs
        self.Y = Y  # Set of all target outputs
        self.y = [] # Set of all actual outputs
        self.w = w  # Set of all weights
        self.bias = 1
        self.threshold = threshold
    
    """Returns sum of each weight/input"""
    def summation(self):
        sum = 0
        self.bias = 0
        self.y = []
        for idx, xi in enumerate(self.X):
            sum += xi * self.w[idx]
            self.y.append(sum)
        self.bias -= sum
        return sum

    """Stores weighted input and passes to activation function, returns result"""
    def calculate_weighted_input(self):
        weighted_input = self.summation() + self.bias
        return self.activation_function(weighted_input)           

    """Step function that checks if value is over threshold"""
    def activation_function(self, x):
        return 1 if x > self.threshold else 0

    """Updates w if threshold not met"""
    def update_w(self):
        updated_w = []
        for idx, xi in enumerate(self.w):
            # wi <-- wi + Δwi
            # Δwi = lr(target - actual)xi
            new_weight = xi + (self.lr * (self.Y[idx] - self.y[idx]) * self.X[idx])
            updated_w.append(new_weight)
        self.w = updated_w

