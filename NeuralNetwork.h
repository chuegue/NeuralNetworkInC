#ifndef NEURALNETWORK
#define NEURALNETWORK
#include "Matrix.h"

typedef struct
{
    int n_layers;
    int *size_layers;
    Matrix **V;
    Matrix **Z;
    Matrix **W;
    Matrix **b;
} NeuralNetwork;

NeuralNetwork *Init_NN(int n_layers, int *size_layers, float gen_func(int, int));
NeuralNetwork *Forward_Propagation(NeuralNetwork *NN, float *inputs, int inputs_size, float hidden_activation_fun(float), float output_activation_fun(float));
NeuralNetwork *Back_Propagation(NeuralNetwork *NN, float *expected_outputs, int outputs_size, float alpha, float ddx_output(float a), float ddx_hidden(float a));
void PrintOutputs_NN(NeuralNetwork *nn);
void PrintLastWeights_NN(NeuralNetwork *nn);
void Print_NN(NeuralNetwork *nn);
void Free_NN(NeuralNetwork *NN);

#endif