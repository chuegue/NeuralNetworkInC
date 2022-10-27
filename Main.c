#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "NeuralNetwork.h"

float gen(int a, int b)
{
    return 1; //(rand() % 11) - 5;
}

float ReLU(float a)
{
    return (a > 0 ? a : 0);
}

float teste(float a)
{
    return a;
}

float ddx(float a)
{
    return (float)1;
}

float sigmoid(float a)
{
    return (1 / (1 + powf(M_E,  -a)));
}

float sigmoid_prime(float a)
{
    return (sigmoid(a) * (1 - sigmoid(a)));
}

float Heavyside(float a)
{
    return (a > 0 ? 1 : 0);
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    // float *inputs = (float *)malloc(2 * sizeof(float)), *outputs = (float *)malloc(2 * sizeof(float));
    //  for (int i = 0; i < 2; i++)
    //  {
    //      inputs[i] = gen(0, 0);
    //      outputs[i] = gen(0, 0);
    //  }
    float inputs[] = {1, 1};
    float outputs[] = {-5, -50};
    int sizelayers[] = {2, 2};
    NeuralNetwork *nn = Init_NN(2, sizelayers, gen);
    Print_NN(nn);
    for (int i = 0; i < 5; i++)
    {
        printf("---FORWARD PROPAGATION---\n");
        nn = Forward_Propagation(nn, inputs, 2, ReLU, ReLU);
        // Print_NN(nn);
        PrintOutputs_NN(nn);
        printf("---BACK PROPAGATION---\n");
        nn = Back_Propagation(nn, outputs, 2, 0.1, teste,Heavyside);
        Print_NN(nn);
    }

    Free_NN(nn);
    return 0;
}