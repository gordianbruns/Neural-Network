''' File:       main.py
 *  Purpose:    Creates a neural network
 *
 *  Input:      filename, epochs, learning_rate
 *  Output:     printed loss and for every 1000 epochs; printed final dictionary, which represents the neural network
 *
 *  Usage:      python main.py [file]<String> [epochs]<int > 0> [learning_rate]<float (between 0.0 and 1.0)>
 *
'''

import sys              # for reading the command line arguments
import numpy as np      # for numpy arrays
import pandas as pd     # for saving the file as a dataframe


def main(argv):
    error_message = "Correct usage: python main.py [file]<String> [epochs]<int > 0> "\
                    "[learning_rate]<float (between 0.0 and 1.0)>"
    if len(sys.argv) != 4:
        print(error_message)
        exit(-1)
    try:
        file_name = sys.argv[1]

        epochs = int(sys.argv[2])
        if epochs <= 0:
            print(error_message)
            exit(-1)

        learning_rate = float(sys.argv[3])
        if learning_rate < 0.0 or learning_rate > 1.0:
            print(error_message)
            exit(-1)

    except IndexError:
        print(error_message)
        exit(-1)

    result = model_file(file_name, 2, 2, 1, epochs, learning_rate)
    print(result)
'''  main  '''


''' Function:    xor
 *  Purpose:     determines the output
 *  Input args:  x<binary num>, y<binary num>
 *  Return val:  0 or 1
'''
def xor(x, y):
    if x == 0 and y == 0 or x == 1 and y == 1:
        return 0
    else:
        return 1
'''  xor  '''


''' Function:    sigmoid
 *  Purpose:     normalizes a given value
 *  Input args:  x<float>
 *  Return val:  normalized value<float>
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''  sigmoid  '''


''' Function:    init_weights_biases
 *  Purpose:     initializes the dictionary
 *  Input args:  num_input_nodes<int>, num_hidden_nodes<int>, num_output_nodes<int>
 *  Return val:  parameter_dictionary<dictionary>
'''
def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
    parameter_dictionary = dict()
    parameter_dictionary["hidden_biases"] = np.zeros((num_hidden_nodes, 1))
    parameter_dictionary["output_biases"] = np.zeros((num_output_nodes, num_output_nodes))
    parameter_dictionary["hidden_weights"] = np.random.randn(num_input_nodes, num_input_nodes)
    parameter_dictionary["output_weights"] = np.random.randn(num_output_nodes, num_hidden_nodes)
    return parameter_dictionary
'''  init_weights_biases  '''


''' Function:    read_file_to_array
 *  Purpose:     reads the file and creates a features, labels, and headers array
 *  Input args:  file_name<String>
 *  Return val:  features<np.array>, labels<np.array>, headers<np.array>
'''
def read_file_to_array(file_name):
    try:
        df = pd.read_table(file_name, delim_whitespace=True)
    except FileNotFoundError:
        print("Error opening the file! Exiting . . .")
        exit(-1)
    header_list = list()
    feature_list = list()
    labels_list = list()
    header = list(df)
    for head in header:
        header_list.append([head])
    headers = np.array(header_list)
    for index in range(len(header_list)):
        key = header[index]
        if index == len(header_list) - 1:
            labels_list = list(df[key])
        else:
            feature_list.append(list(df[key]))
    labels = np.array([labels_list]).astype(np.float)
    features = np.array(feature_list).astype(np.float)
    return features, labels, headers
'''  read_file_to_array  '''


''' Function:    forward_propagate
 *  Purpose:     forward propagates through the nodes
 *  Input args:  feature<np.array>, current_weights_biases<dictionary>
 *  Return val:  output_vals<dictionary>
'''
def forward_propagate(feature, current_weights_biases):
    hidden_layer_values = np.dot(current_weights_biases["hidden_weights"], feature)\
                          + current_weights_biases["hidden_biases"]
    hidden_layer_outputs = sigmoid(hidden_layer_values)
    output_values = np.dot(current_weights_biases["output_weights"], hidden_layer_outputs)\
                    + current_weights_biases["output_biases"]
    output_layer_outputs = sigmoid(output_values)
    output_vals = {"hidden_layer_outputs": hidden_layer_outputs, "output_layer_outputs": output_layer_outputs}
    return output_vals
'''  forward_propagate  '''


''' Function:    find_loss
 *  Purpose:     finds the loss the neural network produces
 *  Input args:  output_layer_outputs<dictionary>, labels<np.array>
 *  Return val:  loss<float>
'''
def find_loss(output_layer_outputs, labels):
    num_examples = labels.shape[1]
    loss = (-1 / num_examples) * np.sum(np.multiply(labels, np.log(output_layer_outputs["output_layer_outputs"])) +
                                        np.multiply(1 - labels, np.log(1 - output_layer_outputs["output_layer_outputs"])))
    return loss
'''  find_loss  '''


''' Function:    backprop
 *  Purpose:     backpropagates through the nodes to get the gradients of each weight and bias
 *  Input args:  feature_array<np.array>, labels<np.array>, output_vals<dictionary>, weights_biases_dict<dictionary>
 *  Return val:  gradients<dictionary>
'''
def backprop(feature_array, labels, output_vals, weights_biases_dict, verbose=False):
    if verbose:
        print()
    # We get the number of examples by looking at how many total
    # labels there are. (Each example has a label.)
    num_examples = labels.shape[1]

    # These are the outputs that were calculated by each
    # of our two layers of nodes that calculate outputs.
    hidden_layer_outputs = output_vals["hidden_layer_outputs"]
    output_layer_outputs = output_vals["output_layer_outputs"]

    # These are the weights of the arrows coming into our output
    # node from each of the hidden nodes. We need these to know
    # how much blame to place on each hidden node.
    output_weights = weights_biases_dict["output_weights"]

    # This is how wrong we were on each of our examples, and in
    # what direction. If we have four training examples, there
    # will be four of these.
    # This calculation works because we are using binary cross-entropy,
    # which produces a fairly simply calculation here.
    raw_error = output_layer_outputs - labels
    if verbose:
        print("raw_error", raw_error)

    # This is where we calculate our gradient for each of the
    # weights on arrows coming into our output.
    output_weights_gradient = np.dot(raw_error, hidden_layer_outputs.T) / num_examples
    if verbose:
        print("output_weights_gradient", output_weights_gradient)

    # This is our gradient on the bias. It is simply the
    # mean of our errors.
    output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True) / num_examples
    if verbose:
        print("output_bias_gradient", output_bias_gradient)

    # We now calculate the amount of error to propegate back to our hidden nodes.
    # First, we find the dot product of our output weights and the error
    # on each of four training examples. This allows us to figure out how much,
    # for each of our training examples, each hidden node contributed to our
    # getting things wrong.
    blame_array = np.dot(output_weights.T, raw_error)
    if verbose:
        print("blame_array", blame_array)

    # hidden_layer_outputs is the actual values output by our hidden layer for
    # each of the four training examples. We square each of these values.
    hidden_outputs_squared = np.power(hidden_layer_outputs, 2)
    if verbose:
        print("hidden_layer_outputs", hidden_layer_outputs)
        print("hidden_outputs_squared", hidden_outputs_squared)

    # We now multiply our blame array by 1 minus the squares of the hidden layer's
    # outputs.
    propagated_error = np.multiply(blame_array, 1 - hidden_outputs_squared)
    if verbose:
        print("propagated_error", propagated_error)

    # Finally, we compute the magnitude and direction in which we
    # should adjust our weights and biases for the hidden node.
    hidden_weights_gradient = np.dot(propagated_error, feature_array.T) / num_examples
    hidden_bias_gradient = np.sum(propagated_error, axis=1, keepdims=True) / num_examples
    if verbose:
        print("hidden_weights_gradient", hidden_weights_gradient)
        print("hidden_bias_gradient", hidden_bias_gradient)

    # A dictionary that stores all of the gradients
    # These are values that track which direction and by
    # how much each of our weights and biases should move
    gradients = {"hidden_weights_gradient": hidden_weights_gradient,
                 "hidden_bias_gradient": hidden_bias_gradient,
                 "output_weights_gradient": output_weights_gradient,
                 "output_bias_gradient": output_bias_gradient}

    return gradients
'''  backprop  '''


''' Function:    update_weights_biases
 *  Purpose:     adjusts the weights and biases
 *  Input args:  parameter_dictionary<dictionary>, gradients<dictionary>, learning_rate<float>
 *  Return val:  updated_parameters<dictionary>
'''
def update_weights_biases(parameter_dictionary, gradients, learning_rate):
    new_hidden_weights = parameter_dictionary["hidden_weights"] - learning_rate * gradients["hidden_weights_gradient"]

    new_hidden_biases = parameter_dictionary["hidden_biases"] - learning_rate * gradients["hidden_bias_gradient"]

    new_output_weights = parameter_dictionary["output_weights"] - learning_rate * gradients["output_weights_gradient"]

    new_output_biases = parameter_dictionary["output_biases"] - learning_rate * gradients["output_bias_gradient"]

    updated_parameters = {"hidden_weights": new_hidden_weights,
                          "hidden_biases": new_hidden_biases,
                          "output_weights": new_output_weights,
                          "output_biases": new_output_biases}
    return updated_parameters
'''  update_weights_biases  '''


''' Function:    model_file
 *  Purpose:     executes the whole process
 *  Input args:  file_name<String>, num_inputs<int>, num_hiddens<int>, num_outputs<int>, epochs<int>, learning_rate<float>
 *  Return val:  updated_dictionary<dictionary>
'''
def model_file(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
    features, labels, headers = read_file_to_array(file_name)

    dictionary = init_weights_biases(num_inputs, num_hiddens, num_outputs)

    for epoch in range(epochs):
        output_values = forward_propagate(features, dictionary)

        if epoch % 1000 == 0:
            loss = find_loss(output_values, labels)
            print("epoch =", epoch, "loss =", loss)

        gradients = backprop(features, labels, output_values, dictionary)

        dictionary = update_weights_biases(dictionary, gradients, learning_rate)

    updated_dictionary = dictionary

    return updated_dictionary
'''  model_file  '''


if __name__ == "__main__":
    main(sys.argv[:3])
