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
} NeuralNetwork;

NeuralNetwork *Init_NN(int n_layers, int *size_layers, float gen_func(int, int));
NeuralNetwork *Forward_Propagation(NeuralNetwork *NN, float *inputs, int inputs_size, float hidden_activation_fun(float), float output_activation_fun(float));
void Free_NN(NeuralNetwork *NN);

#endif