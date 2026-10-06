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
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0; // считаем сумму попарных элементов для каждого элемента матрицы C
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[j * K + k];  // A[k * M + i] = A[i][k] (column-major, M — число строк). 
            }
            C[j * M + i] = sum;
        }
    }
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
    const int N = 2;   // ← маленькая матрица

    // Выделяем
    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);
    double* C = alloc_matrix(N);

    // Задаём ВРУЧНУЮ известные значения (column-major!)
    // A = [[1, 2],       в column-major: [1, 3, 2, 4]
    //      [3, 4]]
    A[0] = 1; A[1] = 3; A[2] = 2; A[3] = 4;

    // B = [[5, 6],       в column-major: [5, 7, 6, 8]
    //      [7, 8]]
    B[0] = 5; B[1] = 7; B[2] = 6; B[3] = 8;

    // Обнуляем C
    zero_matrix(C, N);

    // Считаем
    dgemm_seq(N, N, N, A, B, C);

    // Выводим результат
    printf("C[0][0] = %f (waiting for 19)\n", C[0 * N + 0]);
    printf("C[0][1] = %f (waiting for 22)\n", C[1 * N + 0]);
    printf("C[1][0] = %f (waiting for 43)\n", C[0 * N + 1]);
    printf("C[1][1] = %f (waiting for 50)\n", C[1 * N + 1]);

    free(A);
    free(B);
    free(C);
    return 0;
}