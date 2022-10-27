#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "NeuralNetwork.h"

float gen(int a, int b)
{
    return 1;//(rand() % 11) + 1;
}

float ReLU(float a)
{
    return (a > 0 ? a : 0);
}

float Heavyside(float a)
{
    return (a > 0 ? 1 : 0);
}

float sigmoid(float a)
{
    return (1 / (1 + powf(M_E, -a)));
}

float sigmoid_prime(float a)
{
    return (sigmoid(a) * (1 - sigmoid(a)));
}

float linear(float a)
{
    return a;
}

float ddx_linear(float a)
{
    return 1;
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
    float inputs[] = {1};
    float outputs[] = {-5};
    int sizelayers[] = {1, 1};
    NeuralNetwork *nn = Init_NN(2, sizelayers, gen);
    Print_NN(nn);
    for (int i = 0; i < 5; i++)
    {
        // printf("---FORWARD PROPAGATION---\n");
        nn = Forward_Propagation(nn, inputs, 1, ReLU, linear);
        // Print_NN(nn);
        // PrintOutputs_NN(nn);
        // printf("---BACK PROPAGATION---\n");
        nn = Back_Propagation(nn, outputs, 1, 0.1, Heavyside, ddx_linear);
         Print_NN(nn);
        // sleep(1);
    }
    Print_NN(nn);
    PrintOutputs_NN(nn);
    Free_NN(nn);
    return 0;
}