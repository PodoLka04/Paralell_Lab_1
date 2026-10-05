// ============================================================
// Домашнее задание №1 (OpenMP)
// Параллельное умножение матриц (DGEMM)
// Column-major, интерфейс BLAS
// ============================================================

#include <cstdio>      // printf
#include <cstdlib>     // malloc, free, rand, srand
#include <cmath>       // fabs
#include <omp.h>       // OpenMP

// ============================================================
// Утилиты
// ============================================================

// Выделить память под матрицу n x n (column-major)
double* alloc_matrix(int n)
{
    double* M = (double*)malloc(sizeof(double) * n * n);
    if (M == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return M;
}

// Заполнить матрицу случайными числами [0, 1]
void init_matrix(double* M, int n)
{
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)rand() / RAND_MAX;
    }
}

// Обнулить матрицу
void zero_matrix(double* M, int n)
{
    for (int i = 0; i < n * n; i++) {
        M[i] = 0.0;
    }
}

// Сравнить две матрицы — вернуть максимальное отличие
double compare_matrices(double* A, double* B, int n)
{
    double max_diff = 0.0;
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// ============================================================
// Задание 1: последовательная версия (column-major)
// ============================================================
void dgemm_seq(int M, int N, int K, double* A, double* B, double* C)
{
    // TODO: реализовать
}

// ============================================================
// Задание 1: параллельная версия (OpenMP)
// ============================================================
void blas_dgemm(int M, int N, int K, double* A, double* B, double* C)
{
    // TODO: реализовать
}

// ============================================================
// main
// ============================================================
int main()
{
    const int N = 4;   // маленький размер для проверки

    srand(42);

    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);
    double* C = alloc_matrix(N);

    init_matrix(A, N);
    init_matrix(B, N);
    zero_matrix(C, N);

    // Пока ничего не вызываем — TODO
    printf("Matrix allocated. N = %d\n", N);

    free(A);
    free(B);
    free(C);

    return 0;
}