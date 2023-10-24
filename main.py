import numpy as np

class SimpleResidualNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden1 = np.random.randn(input_size, hidden1_size)
        self.biases_hidden1 = np.zeros((1, hidden1_size))
        self.weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size)
        self.biases_hidden2 = np.zeros((1, hidden2_size))
        self.weights_hidden2_output = np.random.randn(hidden2_size, output_size)
        self.biases_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Input to hidden layer 1
        self.hidden1_input = np.dot(X, self.weights_input_hidden1) + self.biases_hidden1
        self.hidden1_output = self.sigmoid(self.hidden1_input)

        # Hidden layer 1 to hidden layer 2
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.biases_hidden2
        self.hidden2_output = self.sigmoid(self.hidden2_input + self.hidden1_input)

        # Hidden layer 2 to output
        self.output_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.biases_output
        self.predicted_output = self.sigmoid(self.output_input)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Calculate loss
        loss = y - self.predicted_output

        # Backpropagation
        d_output = loss * self.sigmoid_derivative(self.predicted_output)
        d_hidden2 = d_output.dot(self.weights_hidden2_output.T) * self.sigmoid_derivative(self.hidden2_output)
        d_hidden1 = d_hidden2.dot(self.weights_hidden1_hidden2.T) * self.sigmoid_derivative(self.hidden1_output)

        # Update weights and biases
        self.weights_hidden2_output += self.hidden2_output.T.dot(d_output) * learning_rate
        self.biases_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_hidden1_hidden2 += self.hidden1_output.T.dot(d_hidden2) * learning_rate
        self.biases_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate
        self.biases_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Sample dataset (XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden1_size = 4
    hidden2_size = 4
    output_size = 1

    mlp = MLP(input_size, hidden1_size, hidden2_size, output_size)
    mlp.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the trained model
    predictions = mlp.forward(X)
    print("Final predictions:")
    print(predictions)