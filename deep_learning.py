import tensorflow as tf
import numpy as np


class NeuralNet(tf.keras.Sequential):
    """A tensorflow Sequential model object"""
    def __init__(self, kernel_initializer='glorot_uniform'):
        super().__init__()
        # Input layer is not explicit but it would be: tf.keras.Input(shape=(64,0), batch_size=1)
        # e.g. tf.random.normal(shape=(1, 64)) can be passed to self.call
        if isinstance(kernel_initializer, str):
            self.dense1 = tf.keras.layers.Dense(16, activation="relu", name='dense1', kernel_initializer=kernel_initializer)
            self.dense2 = tf.keras.layers.Dense(8, activation="relu", name='dense2', kernel_initializer=kernel_initializer)
            self.dense3 = tf.keras.layers.Dense(1024, activation="softmax", name='dense3', kernel_initializer=kernel_initializer)
        elif isinstance(kernel_initializer, list):
            self.dense1 = tf.keras.layers.Dense(16, activation="relu", name='dense1',
                                                kernel_initializer=tf.keras.initializers.constant(kernel_initializer[0]))
            self.dense2 = tf.keras.layers.Dense(8, activation="relu", name='dense2',
                                                kernel_initializer=tf.keras.initializers.constant(kernel_initializer[1]))
            self.dense3 = tf.keras.layers.Dense(1024, activation="softmax", name='dense3',
                                                kernel_initializer=tf.keras.initializers.constant(kernel_initializer[2]))
        self.dense_layers = [self.dense1, self.dense2, self.dense3]
        self.player = None
        self.board_shape = (1, 64)
        self.build(input_shape=self.board_shape)

    def call(self, inputs, **kwargs):
        # x = self.dense1(inputs)
        # x1 = self.dense2(x)
        # y = self.dense3(x1)

        y = inputs
        for layer in self.dense_layers:
            y = layer(y)
        return y

    def get_weights(self):
        # silent_build = self.call(tf.random.normal(shape=(1, 64)))
        return {self.dense1.name: self.dense1.weights[0], self.dense2.name: self.dense2.weights[0], self.dense3.name: self.dense3.weights[0]}

    def forward_pass(self, inputs):
        """call the sequential model and then cancel (zero out) the illegal steps"""
        if isinstance(inputs, list):
            inputs = np.array(inputs).reshape(self.board_shape)

        if type(inputs).__module__ == np.__name__:
            inputs = inputs.reshape(self.board_shape)

        # convert to tensor
        inputs = tf.constant(inputs)

        # call
        scores_tensor = self.call(inputs)
        shp = scores_tensor.numpy().shape   # shp = 1, 1024
        scores_numpy = scores_tensor[0].numpy()

        all_steps = self.player.steps_encoded.copy()
        legal_steps = self.player.encode_legal_steps()
        illegal_indices = []
        for step in all_steps:
            try:
                # the step is legal
                legal_steps.index(step)
            except ValueError:
                # the step is illegal
                illegal_indices.append(all_steps.index(step))

        # Set the score for illegal steps to zero
        scores_numpy[illegal_indices] = 0

        # Introduce some randomness so that it does not get stuck
        r = np.random.normal(size=1)
        if r > -1.64:  # around 0.05 probability that r is smaller
            # Choose step with the maximal score
            best_step = all_steps[np.argmax(scores_numpy)]
        else:
            # Choose the step with second largest score
            second_best_step = all_steps[np.argpartition(scores_numpy.flatten(), -2)[-2]]
            best_step = second_best_step

        # Decode step
        x, y, id = self.player.decode_step(best_step)

        # return selected piece and new position
        p = self.player.get_piece(id=id)
        new_pos = np.array([x, y])

        return p, new_pos


if __name__ == '__main__':
    nn = NeuralNet()
    my_inputs = tf.random.normal(shape=(1, 64))
    scores = nn.call(inputs=my_inputs)
    print(nn.get_weights())



