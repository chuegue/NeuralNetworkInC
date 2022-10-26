#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Matrix.h"
#include "NeuralNetwork.h"

float zeros(float a, float b)
{
    return (float)0;
}

NeuralNetwork *Init_NN(int n_layers, int *size_layers)
{
    NeuralNetwork *NN = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    NN->size_layers = (int *)malloc(n_layers * sizeof(int));
    memcpy(NN->size_layers, size_layers, n_layers * sizeof(int));
    NN->V = (Matrix **)malloc(n_layers * sizeof(Matrix *));
    NN->W = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    NN->V = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    for (int i = 0; i < n_layers; i++)
    {
        NN->V[i] = Init_Matrix(size_layers[i], 1, zeros);
    }
    for (int i = 0; i < n_layers - 1; i++)
    {
        NN->W[i] = Init_Matrix(size_layers[i], size_layers[i + 1], zeros);
        NN->b[i] = Init_Matrix(size_layers[i + 1], 1, zeros);
    }
    return NN;
}

