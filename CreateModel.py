import tensorflow as tf


def create_model(size_x, size_t, neurons_per_layer):

        # Inputs layer
        branch_input_u0 = tf.keras.Input(shape=(size_x), name='branch_input_u0')
        branch_input_p0 = tf.keras.Input(shape=(size_x), name='branch_input_p0')
        branch_input_U = tf.keras.Input(shape=(size_t * size_x), name='branch_input_U')
        branch_input_P = tf.keras.Input(shape=(size_t * size_x), name='branch_input_P')
        trunk_input = tf.keras.Input(shape=(2,), name='trunk_input')

        # DeepONet (u)
        branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer0_u0')(branch_input_u0)
        branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer0_p0')(branch_input_p0)
        branch_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer0_U')(branch_input_U)
        branch_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer0_P')(branch_input_P)
        trunk = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_trunk_layer0')(trunk_input)
        for i in range(4):
            branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'u_net_branch_layer{i+1}_u0')(branch_u0)
            branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'u_net_branch_layer{i+1}_p0')(branch_p0)
            branch_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'u_net_branch_layer{i+1}_U')(branch_U)
            branch_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'u_net_branch_layer{i+1}_P')(branch_P)
            trunk = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'u_net_trunk_layer{i+1}')(trunk)
        branch_output_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer10_u0')(branch_u0)
        branch_output_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer10_p0')(branch_p0)
        branch_output_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer10_U')(branch_U)
        branch_output_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_branch_layer10_P')(branch_P)
        trunk_output = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='u_net_trunk_layer10')(trunk)
        u_output = tf.reduce_sum(tf.multiply(branch_output_u0, tf.multiply(branch_output_p0, tf.multiply(tf.multiply(branch_output_U, branch_output_P), trunk_output))), axis=1, keepdims=True)

        # DeepONet (p)
        branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer0_u0')(branch_input_u0)
        branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer0_p0')(branch_input_p0)
        branch_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer0_U')(branch_input_U)
        branch_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer0_P')(branch_input_P)
        trunk = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_trunk_layer0')(trunk_input)
        for i in range(4):
            branch_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'p_net_branch_layer{i+1}_u0')(branch_u0)
            branch_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'p_net_branch_layer{i+1}_p0')(branch_p0)
            branch_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'p_net_branch_layer{i+1}_U')(branch_U)
            branch_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'p_net_branch_layer{i+1}_P')(branch_P)
            trunk = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name=f'p_net_trunk_layer{i+1}')(trunk)
        branch_output_u0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer10_u0')(branch_u0)
        branch_output_p0 = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer10_p0')(branch_p0)
        branch_output_U = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer10_U')(branch_U)
        branch_output_P = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_branch_layer10_P')(branch_P)
        trunk_output = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', name='p_net_trunk_layer10')(trunk)
        p_output = tf.reduce_sum(tf.multiply(branch_output_u0, tf.multiply(branch_output_p0, tf.multiply(tf.multiply(branch_output_U, branch_output_P), trunk_output))), axis=1, keepdims=True)

        # Model
        model = tf.keras.models.Model(inputs=[branch_input_u0, branch_input_p0, branch_input_U, branch_input_P, trunk_input], outputs=[u_output, p_output])

        return model
