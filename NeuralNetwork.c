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
    NN->Z = (Matrix **)malloc(n_layers * sizeof(Matrix *));
    NN->W = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    NN->b = (Matrix **)malloc((n_layers - 1) * sizeof(Matrix *));
    for (int i = 0; i < n_layers; i++)
    {
        NN->V[i] = Init_Matrix(size_layers[i], 1, gen_func);
        NN->Z[i] = Init_Matrix(size_layers[i], 1, gen_func);
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
        NN->V[0]->data[i][0] = NN->Z[0]->data[i][0] = inputs[i];
    }
    for (int i = 1; i < NN->n_layers; i++)
    {
        // V[i] = f(W[i-1]T * V[i-1] + b[i-1])
        Matrix *WTranspose = Transpose(NN->W[i - 1]);
        Matrix *WTV = Multiply(WTranspose, NN->V[i - 1]);
        Matrix *Z = Addition(WTV, NN->b[i - 1]);
        Matrix *V;
        CopyContents_Matrix(NN->Z[i], Z);
        if (i != NN->n_layers - 1)
        {
            V = ApplyFunc_ElementWise_Matrix(Z, hidden_activation_fun);
        }
        else
        {
            V = ApplyFunc_ElementWise_Matrix(Z, output_activation_fun);
        }
        CopyContents_Matrix(NN->V[i], V);
        Free_Matrix(Z);
        Free_Matrix(V);
        Free_Matrix(WTV);
        Free_Matrix(WTranspose);
    }
    return NN;
}

float zeros(int a, int b)
{
    return (float)0;
}

NeuralNetwork *Back_Propagation(NeuralNetwork *NN, float *expected_outputs, int outputs_size, float alpha, float ddx_hidden(float a), float ddx_output(float a))
{
    //   dE/dW = (dE/dV)(dV/dZ)(dZ/dW)
    assert(outputs_size == NN->size_layers[NN->n_layers - 1]);
    Matrix *expected_outputs_matrix = Init_Matrix(outputs_size, 1, zeros);
    for (int i = 0; i < outputs_size; i++)
    {
        expected_outputs_matrix->data[i][0] = expected_outputs[i];
    }
    Matrix *VminusY = Subtraction(NN->V[NN->n_layers - 1], expected_outputs_matrix);
    // Matrix *VminusY2 = EntryWise_Multiply(VminusY, VminusY);
    Matrix *dvdz = ApplyFunc_ElementWise_Matrix(NN->Z[NN->n_layers - 1], ddx_output);
    Matrix *VT = Transpose(NN->V[NN->n_layers - 2]);
    Matrix *DELTA = EntryWise_Multiply(VminusY, dvdz);
    Matrix *dEdW = Multiply(DELTA, VT);
    Matrix *alpha_dEdW = Scalar_Multiply(dEdW, alpha);
    Matrix *alpha_dEdW_T = Transpose(alpha_dEdW);
    Matrix *delta_w = Subtraction(NN->W[NN->n_layers - 2], alpha_dEdW_T);
    CopyContents_Matrix(NN->W[NN->n_layers - 2], delta_w);
    Matrix *alpha_dEdb = Scalar_Multiply(DELTA, alpha);
    Matrix *delta_b = Subtraction(NN->b[NN->n_layers - 2], alpha_dEdb);
    CopyContents_Matrix(NN->b[NN->n_layers - 2], delta_b);
    Free_Matrix(expected_outputs_matrix);
    Free_Matrix(VminusY);
    // Free_Matrix(VminusY2);
    Free_Matrix(dvdz);
    Free_Matrix(VT);
    Free_Matrix(DELTA);
    Free_Matrix(dEdW);
    Free_Matrix(alpha_dEdb);
    Free_Matrix(alpha_dEdW);
    Free_Matrix(alpha_dEdW_T);
    Free_Matrix(delta_w);
    Free_Matrix(delta_b);
    return NN;
}

void PrintOutputs_NN(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->size_layers[nn->n_layers - 1]; i++)
    {
        printf("Output %i: %f\n", i + 1, nn->V[nn->n_layers - 1]->data[i][0]);
    }
    printf("\n");
}

void PrintLastWeights_NN(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->W[nn->n_layers - 2]->rows; i++)
    {
        for (int j = 0; j < nn->W[nn->n_layers - 2]->cols; j++)
        {
            printf("Weight %i-%i: %f\n", i + 1, j + 1, nn->W[nn->n_layers - 2]->data[i][j]);
        }
    }
    printf("\n");
}

void Print_NN(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->n_layers; i++)
    {
        printf("Z[%i]:\n", i);
        Print_Matrix(nn->Z[i]);
        printf("V[%i]:\n", i);
        Print_Matrix(nn->V[i]);
        if (i < nn->n_layers - 1)
        {
            printf("Weights[%i]:\n", i);
            Print_Matrix(nn->W[i]);
            printf("Biases[%i]:\n", i);
            Print_Matrix(nn->b[i]);
        }
    }
}

void Free_NN(NeuralNetwork *NN)
{
    free(NN->size_layers);
    for (int i = 0; i < NN->n_layers; i++)
    {
        Free_Matrix(NN->V[i]);
        Free_Matrix(NN->Z[i]);
    }
    for (int i = 0; i < NN->n_layers - 1; i++)
    {
        Free_Matrix(NN->W[i]);
        Free_Matrix(NN->b[i]);
    }
    free(NN->V);
    free(NN->W);
    free(NN->Z);
    free(NN->b);
    free(NN);
}