README
------------------
Gordian Bruns

CS365

Lab D - Neural Networks
------------------

I. File List
 - main.py  # contains xor, sigmoid, init_weights_biases, read_file_to_array, forward_propagate, find_loss, backprop, update_weights_biases, model_file


II. Usage

The program takes three command line arguments:
  - filename
  - epochs
  - learning_rate

Also, epochs must be an integer greater than 0 and learning_rate must be a float between 0.0 and 1.0.

The program will then automatically create a neural network and train it based on the given file for the nnumber of epochs long with the given learning rate.
After every 1000 epochs, it will print the current loss and the current epoch.
Eventually, it will print the final dictionary, which withholds the weights and biases of each node.

To run the program you must be in the directory of the files and type the following into the command line:
python main.py [filename] [epochs] [learning_rate]

Note that the file we use to train the program must be in the same directory as the program.
