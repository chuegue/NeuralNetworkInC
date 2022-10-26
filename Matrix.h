#ifndef _MATRIX_
#define _MATRIX_

typedef struct
{
    float **data;
    int rows;
    int cols;
} Matrix;

Matrix *Init_Matrix(int rows, int cols, float gen_func(int, int));
Matrix *EntryWise_Multiply(Matrix *a, Matrix *b);
Matrix *Multiply(Matrix *a, Matrix *b);
Matrix *Scalar_Multiply(Matrix *a, int m);
Matrix *Addition(Matrix *a, Matrix *b);
Matrix *Scalar_Addition(Matrix *a, int m);
Matrix *Subtraction(Matrix *a, Matrix *b);
Matrix *Transpose(Matrix *a);
double Determinant(Matrix *a);
Matrix *Inverse(Matrix *input);
void Print_Matrix(Matrix *a);
void Free_Matrix(Matrix *a);

#endif