#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Matrix.h"

// initializes a rows x cols Matrix struct with the desired values based on the coordinates in the matrix given by gen_func
Matrix *Init_Matrix(int rows, int cols, float gen_func(int, int))
{
    assert(rows > 0 && cols > 0);
    Matrix *new = (Matrix *)malloc(sizeof(Matrix));
    assert(new != NULL);
    new->rows = rows;
    new->cols = cols;
    new->data = (float **)malloc(rows * sizeof(float *));
    assert(new->data != NULL);
    for (int i = 0; i < rows; i++)
    {
        new->data[i] = (float *)malloc(cols * sizeof(float));
        assert(new->data[i] != NULL);
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            new->data[i][j] = gen_func(i, j);
        }
    }
    return new;
}

float zero(int a, int b)
{
    return 0;
}

// given matrix a n*k and b k*m returns the resulting n*m matrix multiplication
Matrix *Multiply(Matrix *a, Matrix *b)
{
    assert(a->cols == b->rows);
    Matrix *result = Init_Matrix(a->rows, b->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            for (int k = 0; k < a->cols; k++)
            {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return result;
}

Matrix *EntryWise_Multiply(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols && a->rows == b->rows);
    Matrix *result = Init_Matrix(a->rows, a->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            result->data[i][j] += a->data[i][j] * b->data[i][j];
        }
    }
    return result;
}

// given a matrix and a scalar m returns a matrix where each element is multiplied by the scalar
Matrix *Scalar_Multiply(Matrix *a, int m)
{
    Matrix *result = Init_Matrix(a->rows, a->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            result->data[i][j] = a->data[i][j] * m;
        }
    }
    return result;
}

// given two matrices n*m returns the result of the addition of the two matrices
Matrix *Addition(Matrix *a, Matrix *b)
{
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix *result = Init_Matrix(a->rows, a->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return result;
}

// given a matrix and a scalar m returns a matrix where each element is added by the scalar
Matrix *Scalar_Addition(Matrix *a, int m)
{
    Matrix *result = Init_Matrix(a->rows, a->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            result->data[i][j] = a->data[i][j] + m;
        }
    }
    return result;
}

// given two matrices n*m returns the result of the subtraction of the two matrices
Matrix *Subtraction(Matrix *a, Matrix *b)
{
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix *result = Init_Matrix(a->rows, a->cols, zero);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return result;
}

// given a matrix m*n returns a matrix transpose n*m
Matrix *Transpose(Matrix *a)
{
    Matrix *result = Init_Matrix(a->cols, a->rows, zero);
    for (int i = 0; i < a->cols; i++)
    {
        for (int j = 0; j < a->rows; j++)
        {
            result->data[i][j] = a->data[j][i];
        }
    }
    return result;
}

double Determinant(Matrix *a)
{
    int n = a->cols;
    double **aux = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        aux[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            aux[i][j] = a->data[i][j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        if (aux[i][i] == 0)
        {
            for (int i = 0; i < n; i++)
            {
                free(aux[i]);
            }
            free(aux);
            return 0;
        }
        for (int j = i + 1; j < n; j++)
        {
            double ratio = aux[j][i] / aux[i][i];
            for (int k = 0; k < n; k++)
            {
                aux[j][k] -= ratio * aux[i][k];
            }
        }
    }
    double det = 1;
    for (int i = 0; i < n; i++)
    {
        det *= aux[i][i];
    }
    for (int i = 0; i < n; i++)
    {
        free(aux[i]);
    }
    free(aux);
    return det;
}

Matrix *Inverse(Matrix *input)
{
    assert(input->cols == input->rows && input->cols != 0);
    Matrix *a = Init_Matrix(input->rows, input->cols * 2, zero);
    int n = input->cols;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a->data[i][j] = input->data[i][j];
            if (i == j)
                a->data[i][j + n] = 1;
            else
                a->data[i][j + n] = 0;
        }
    }
    for (int i = 0; i < n; i++)
    {
        assert(a->data[i][i] != 0);
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                float ratio = a->data[j][i] / a->data[i][i];
                for (int k = 0; k < 2 * n; k++)
                {
                    a->data[j][k] -= ratio * a->data[i][k];
                }
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = n; j < 2 * n; j++)
        {
            a->data[i][j] /= a->data[i][i];
        }
        a->data[i][i] /= a->data[i][i];
    }
    return a;
}

// returns matrix a with all its entries as the result of function func
Matrix *ApplyFunc_ElementWise_Matrix(Matrix *a, float func(float))
{
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            a->data[i][j] = func(a->data[i][j]);
        }
    }
    return a;
}

void Print_Matrix(Matrix *a)
{
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            if (a->data[i][j] >= 0)
                printf(" ");
            printf("%.3f ", a->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// a = dest, b = source
void CopyContents_Matrix(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols && a->rows == b->rows);
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            a->data[i][j] = b->data[i][j];
        }
    }
}

void Free_Matrix(Matrix *a)
{
    for (int i = 0; i < a->rows; i++)
        free(a->data[i]);
    free(a->data);
    free(a);
}