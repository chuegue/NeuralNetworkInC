#ifndef NEURALNETWORK
#define NEURALNETWORK
#include "Matrix.h"

typedef struct
{
    int n_layers;
    int *size_layers;
    Matrix **V;
    Matrix **W;
    Matrix **b;
    // int *n_inputs;
    // int *n_outputs;
    // int n_hidden_layers;
    // int *n_nodes_hidden_layer;
    // int n_weights;
    // Matrix **weights;
    // int n_biases;
    // Matrix **biases;
} NeuralNetwork;

#endif