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
// Матрица 
// 1 2
// 3 4 
// Как 1 3 2 4
// 
// Пишем Алгоритм через столбцово-ориентированный вид для оптимальности Spatial Locality
// ============================================================


void dgemm_seq(int M, int N, int K, double* A, double* B, double* C)
{
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            C[j * M + i] = 0.0; // обнуляем весь столбец C (мы сразу ег овесь будем обрабатывать), потому что malloc даёт память с мусором.
        }
        // Считаем
        for (int k = 0; k < K; k++) {
            double b = B[j * K + k];        // одно значение B[k][j]
            for (int i = 0; i < M; i++) {
                C[j * M + i] += A[k * M + i] * b; // считаем сумму попарных элементов для каждого элемента матрицы C тут
            }
        }
    }
}

//для каждого столбца j матрицы C :
//
//  обнулить столбец C[:, j]
//  (все элементы C[0][j], C[1][j], ..., C[M - 1][j] = 0)
//
//  для каждой строки k матрицы B :
//      b = B[k][j]                        ← одно число  (идем по столбцу)
//      для каждой строки i матрицы C :
//          C[i][j] += A[i][k] * b         ← добавляем вклад в C[i][j]  (идем по строкам для A, но сразу для всего столбца C значения получаем)



// ============================================================
// Задание 1: параллельная версия (OpenMP)
// ============================================================
void blas_dgemm(int M, int N, int K, double* A, double* B, double* C)
{
#pragma omp parallel for
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) C[j * M + i] = 0;
        for (int k = 0; k < K; k++) {
            double b = B[j * K + k];
            for (int i = 0; i < M; i++) C[j * M + i] += A[k * M + i] * b;
        }
    }
}

// ============================================================
// main
// ============================================================
int main()
{
    const int N = 1000;

    srand(42);

    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);
    double* C_seq = alloc_matrix(N);
    double* C_par = alloc_matrix(N);

    init_matrix(A, N);
    init_matrix(B, N);
    zero_matrix(C_seq, N);
    zero_matrix(C_par, N);

    // Последовательно
    double t1 = omp_get_wtime();
    dgemm_seq(N, N, N, A, B, C_seq);
    double t2 = omp_get_wtime();

    // Параллельно
    double t3 = omp_get_wtime();
    blas_dgemm(N, N, N, A, B, C_par);
    double t4 = omp_get_wtime();

    // Сравнение
    double diff = compare_matrices(C_seq, C_par, N);

    printf("N = %d\n", N);
    printf("Sequential: %.4f sec\n", t2 - t1);
    printf("Parallel:   %.4f sec\n", t4 - t3);
    printf("Speedup:    %.2fx\n", (t2 - t1) / (t4 - t3));
    printf("Max diff:   %e\n", diff);
    printf("Threads:    %d\n", omp_get_max_threads());

    free(A); free(B); free(C_seq); free(C_par);
    return 0;
}