#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "NeuralNetwork.h"

float gen(int a, int b)
{
    return ((float)rand() * 2.0 / RAND_MAX) - 1.0; //(rand() % 11) + 1;
}

float ReLU(float a)
{
    return (a > 0 ? a : 0);
}

float ReLU_prime(float a)
{
    return (a > 0 ? 1 : 0);
}

float leaky_ReLU(float a)
{
    return (a > 0 ? a : 0.001 * a);
}

float leaky_ReLU_prime(float a)
{
    return (a > 0 ? 1 : 0.001);
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

float linear_prime(float a)
{
    return 1;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    float inputs[] = {1};
    float outputs[] = {1};
    int sizelayers[] = {sizeof(inputs) / sizeof(inputs[0]), 3,3, sizeof(outputs) / sizeof(outputs[0])};
    NeuralNetwork *nn = Init_NN(sizeof(sizelayers) / sizeof(sizelayers[0]), sizelayers, gen);
    // Print_NN(nn);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < sizeof(inputs) / sizeof(inputs[0]); j++)
        {
            inputs[j] = 10.0 * rand() / RAND_MAX;
            outputs[j] = 20.0 * inputs[j];
        }
        nn = Forward_Propagation(nn, inputs, sizeof(inputs) / sizeof(inputs[0]), ReLU, linear);
        // Print_NN(nn);

        nn = Back_Propagation(nn, outputs, sizeof(outputs) / sizeof(outputs[0]), 0.001, ReLU_prime, linear_prime);
    }

    // Print_NN(nn);
    for (int n = 0; n < 11; n++)
    {
        for (int j = 0; j < sizeof(inputs) / sizeof(inputs[0]); j++)
        {
            inputs[j] = (float)n;
            printf("Input %i: %f\n", n + 1, inputs[j]);
        }
        nn = Forward_Propagation(nn, inputs, sizeof(inputs) / sizeof(inputs[0]), ReLU, linear);
        PrintOutputs_NN(nn);
        // Print_NN(nn);
    }
    Print_NN(nn);
    Free_NN(nn);
    return 0;
}