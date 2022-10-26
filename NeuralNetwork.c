#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "Matrix.h"
#include "NeuralNetwork.h"

NeuralNetwork *Init_NN(int n_layers, int *size_layers, float gen_func(int, int))
{
    NeuralNetwork *NN = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    NN->n_layers = n_layers;
    NN->size_layers = (int *)malloc(n_layers * sizeof(int));
    memcpy(NN->size_layers, size_layers, n_layers * sizeof(int));
    NN->V = (Matrix **)malloc(n_layers * sizeof(Matrix *));
    NN->W = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    NN->b = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    for (int i = 0; i < n_layers; i++)
    {
        NN->V[i] = Init_Matrix(size_layers[i], 1, gen_func);
    }
    for (int i = 0; i < n_layers - 1; i++)
    {
        NN->W[i] = Init_Matrix(size_layers[i], size_layers[i + 1], gen_func);
        NN->b[i] = Init_Matrix(size_layers[i + 1], 1, gen_func);
    }
    return NN;
}

NeuralNetwork *Forward_Propagation(NeuralNetwork *NN, float *inputs, int inputs_size, float hidden_activation_fun(float), float output_activation_fun(float))
{
    assert(inputs_size == NN->size_layers[0]);
    // put inputs into V[0]
    for (int i = 0; i < NN->size_layers[0]; i++)
    {
        NN->V[0]->data[i][0] = inputs[i];
    }
    for (int i = 1; i < NN->n_layers; i++)
    {
        // V[i] = f(W[i-1]T * V[i-1] + b[i-1])
        Matrix *WTranspose = Transpose(NN->W[i - 1]);
        Matrix *WTV = Multiply(WTranspose, NN->V[i - 1]);
        Matrix *Z = Addition(WTV, NN->b[i - 1]);
        if (i != NN->n_layers - 1)
            Z = ApplyFunc_ElementWise_Matrix(Z, hidden_activation_fun);
        else
            Z = ApplyFunc_ElementWise_Matrix(Z, output_activation_fun);
        CopyContents_Matrix(NN->V[i], Z);
        Free_Matrix(Z);
        Free_Matrix(WTV);
        Free_Matrix(WTranspose);
    }
    return NN;
}

void Free_NN(NeuralNetwork *NN)
{
    free(NN->size_layers);
    for (int i = 0; i < NN->n_layers; i++)
    {
        Free_Matrix(NN->V[i]);
    }
    for (int i = 0; i < NN->n_layers - 1; i++)
    {
        Free_Matrix(NN->W[i]);
        Free_Matrix(NN->b[i]);
    }
    free(NN->V);
    free(NN->W);
    free(NN->b);
    free(NN);
}