#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"

float gen(int a, int b)
{
    return (rand() % 11) - 5;
}

float ReLU(float a)
{
    return (a > 0 ? a : 0);
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    int n = 20;
    float *inputs = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        inputs[i] = gen(0, 0);
    }

    int sizelayers[] = {n, 10 * (n/20), 5* (n/20), 10* (n/20), 5* (n/20)};
    NeuralNetwork *nn = Init_NN(5, sizelayers, gen);
    Forward_Propagation(nn, inputs, n, ReLU, ReLU);
    Free_NN(nn);
    free(inputs);
    return 0;
}