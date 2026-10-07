#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <omp.h>

// ============================================================
// Параметры эксперимента
// ============================================================

const int NUM_RUNS = 10;
const double EPS = 1e-9;


// ============================================================
// Работа с матрицами
// ============================================================

// Выделение матрицы N x N
double* alloc_matrix(int n)
{
    double* M = (double*)malloc((size_t)n * n * sizeof(double));

    if (M == nullptr) {
        printf("Ошибка выделения памяти\n");
        exit(EXIT_FAILURE);
    }

    return M;
}


// Заполнение матрицы случайными числами
void init_matrix(double* M, int n)
{
    for (int i = 0; i < n * n; i++) {
        M[i] = (double)rand() / RAND_MAX;
    }
}


// Обнуление матрицы
void zero_matrix(double* M, int n)
{
    for (int i = 0; i < n * n; i++) {
        M[i] = 0.0;
    }
}


// ============================================================
// Последовательное умножение
// C = A * B
//
// Матрицы хранятся в column-major:
//
// A(i,k) = A[k * M + i]
// B(k,j) = B[j * K + k]
// C(i,j) = C[j * M + i]
//
// Порядок циклов: j -> k -> i
// ============================================================

void dgemm_seq(
    int M,
    int N,
    int K,
    double* A,
    double* B,
    double* C)
{
    for (int j = 0; j < N; j++) {

        // Адрес начала j-й колонки C
        double* c_col = &C[j * M];

        // C[:,j] = 0
        for (int i = 0; i < M; i++) {
            c_col[i] = 0.0;
        }

        for (int k = 0; k < K; k++) {

            // B(k,j)
            double b = B[j * K + k];

            // Адрес начала k-й колонки A
            double* a_col = &A[k * M];

            for (int i = 0; i < M; i++) {

                // C(i,j) += A(i,k) * B(k,j)
                c_col[i] += a_col[i] * b;
            }
        }
    }
}


// ============================================================
// Параллельное DGEMM
//
// Параллелим внешний цикл по j.
//
// Каждый поток получает свои колонки C.
// Поэтому два потока не записывают один и тот же C.
//
// schedule(static) подходит для регулярного DGEMM.
// ============================================================

void blas_dgemm(
    int M,
    int N,
    int K,
    double* A,
    double* B,
    double* C,
    double& elapsed)
{
    double start;
    double end;

#pragma omp parallel
    {
        // Один поток фиксирует начало.
        // После single есть неявный barrier.    ТУТ МЫ СНАЧАЛ СОЗДАЕМ ПОТОКИ, ПОТОМ ОТСЧЕТ ЗАПУСКАЕМ: Накладные на создание потоков — не часть умножения.
#pragma omp single
        {
            start = omp_get_wtime();
        }

        // Параллельная работа
#pragma omp for schedule(static)
        for (int j = 0; j < N; j++) {

            double* c_col = &C[j * M];

            for (int i = 0; i < M; i++) {
                c_col[i] = 0.0;
            }

            for (int k = 0; k < K; k++) {

                double b = B[j * K + k];

                double* a_col = &A[k * M];

                for (int i = 0; i < M; i++) {
                    c_col[i] += a_col[i] * b;
                }
            }
        }

        // У omp for здесь есть неявный barrier.
        // Поэтому все потоки уже закончили вычисление.

#pragma omp single
        {
            end = omp_get_wtime();
        }
    }

    elapsed = end - start;
}


// ============================================================
// Проверка двух матриц
//
// Сравниваем последовательный и параллельный результаты.
// Возвращаем максимальную абсолютную ошибку.
// ============================================================

double compare_matrices(
    double* A,
    double* B,
    int n)
{
    double max_diff = 0.0;

    for (int i = 0; i < n * n; i++) {

        double diff = fabs(A[i] - B[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}


// ============================================================
// Измерение времени  полседовательного и параллельного DGEMM
//
// NUM_RUNS раз запускаем умножение
// и считаем среднее время.
// 
// T_seq  = среднее 10 sequential запусков
// T_par = среднее 10 parallel запусков
// ============================================================

double measure_seq(
    int M,
    int N,
    int K,
    double* A,
    double* B,
    double* C)
{
    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {

        double start = omp_get_wtime();

        dgemm_seq(M, N, K, A, B, C);

        double end = omp_get_wtime();

        total_time += end - start;
    }

    return total_time / NUM_RUNS;
}


double measure_parallel(
    int M,
    int N,
    int K,
    double* A,
    double* B,
    double* C)
{
    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {

        double elapsed;

        blas_dgemm(
            M,
            N,
            K,
            A,
            B,
            C,
            elapsed
        );

        total_time += elapsed;
    }

    return total_time / NUM_RUNS;
}


// ============================================================
// GFLOPS
//
// Для матричного умножения:
//
// FLOPs = 2 * M * N * K
//
// GFLOPS = FLOPs / time / 1e9
// ============================================================

double calculate_gflops(
    int M,
    int N,
    int K,
    double time)
{
    double flops = 2.0 * M * N * K;

    return flops / time / 1e9;
}


// ============================================================
// Один эксперимент
// ============================================================

void run_test(
    int N,
    int threads,
    double* A,
    double* B,
    double* C_seq,
    double* C_par)
{
    printf("\n========================================\n");
    printf("N = %d, threads = %d\n", N, threads);
    printf("========================================\n");

    // --------------------------------------------------------
    // Устанавливаем количество OpenMP потоков
    // --------------------------------------------------------

    omp_set_num_threads(threads);

    // --------------------------------------------------------
    // Проверка правильности
    // --------------------------------------------------------

    dgemm_seq(
        N,
        N,
        N,
        A,
        B,
        C_seq
    );

    blas_dgemm(
        N,
        N,
        N,
        A,
        B,
        C_par
    );

    double max_diff = compare_matrices(
        C_seq,
        C_par,
        N
    );

    printf("Max difference = %.12e\n", max_diff);

    if (max_diff < EPS) {
        printf("CHECK: OK\n");
    }
    else {
        printf("CHECK: FAILED\n");
    }

    // --------------------------------------------------------
    // Измерение времени
    // --------------------------------------------------------

    double time = measure_parallel(
        N,
        N,
        N,
        A,
        B,
        C_par
    );

    double gflops = calculate_gflops(
        N,
        N,
        N,
        time
    );

    printf("Average time = %.6f sec\n", time);
    printf("GFLOPS       = %.3f\n", gflops);
}


// ============================================================
// main
// ============================================================

int main(int argc, char** argv)
{
    // --------------------------------------------------------
    // Размер матрицы
    //
    // Можно запустить:
    //
    // ./gemm_demo
    // ./gemm_demo 500
    // ./gemm_demo 1000
    // --------------------------------------------------------

    int N = 500;

    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("========================================\n");
    printf("OpenMP DGEMM demo\n");
    printf("========================================\n");

    printf("Matrix size: %d x %d\n", N, N);

    // --------------------------------------------------------
    // Отключаем динамическое изменение числа потоков
    // --------------------------------------------------------

    omp_set_dynamic(0);

    printf("Maximum OpenMP threads: %d\n", omp_get_max_threads());

    // --------------------------------------------------------
    // Выделяем память
    // --------------------------------------------------------

    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);

    double* C_seq = alloc_matrix(N);
    double* C_par = alloc_matrix(N);

    // --------------------------------------------------------
    // Инициализация
    // --------------------------------------------------------

    srand(42);

    init_matrix(A, N);
    init_matrix(B, N);

    zero_matrix(C_seq, N);
    zero_matrix(C_par, N);

    // --------------------------------------------------------
    // Набор потоков для эксперимента
    //
    // Если на ПК меньше 24 потоков, не обязательно
    // запускать все значения.
    // --------------------------------------------------------

    int thread_counts[] = {
        1,
        2,
        4,
        8,
        16,
        24
    };

    int num_thread_tests =
        sizeof(thread_counts) / sizeof(thread_counts[0]);

    // --------------------------------------------------------
    // Запускаем эксперименты
    // --------------------------------------------------------

    for (int p = 0; p < num_thread_tests; p++) {

        int threads = thread_counts[p];

        // Не запускаем больше потоков, чем доступно
        // аппаратных потоков на машине.
        if (threads > omp_get_num_procs()) {
            continue;
        }

        run_test(
            N,
            threads,
            A,
            B,
            C_seq,
            C_par
        );
    }

    // --------------------------------------------------------
    // Освобождаем память
    // --------------------------------------------------------

    free(A);
    free(B);
    free(C_seq);
    free(C_par);

    return 0;
}