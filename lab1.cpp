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
// main
// ============================================================

int main(int argc, char* argv[])
{
    // ========================================================
    // Размеры матриц
    //
    // Без аргумента:
    //     ./gemm   
    //     -> 500, 1000, 1500
    //
    // С аргументом:
    //     ./gemm 200
    //     -> только N = 200
    // ========================================================s

    int matrix_sizes[] = {
        500, 1000, 1500
    };

    int num_sizes = 3;

    // Если передали N через командную строку,
    // запускаем только этот размер.
    if (argc > 1) {

        int custom_N = atoi(argv[1]);

        if (custom_N <= 0) {
            printf("Ошибка: N должно быть положительным числом.\n");
            printf("Пример: %s 200\n", argv[0]);
            return 1;
        }

        matrix_sizes[0] = custom_N;
        num_sizes = 1;
    }


    srand(42);
    omp_set_dynamic(0);


    printf("OpenMP DGEMM demo\n");

    if (num_sizes == 1) {
        printf("Matrix size: %d x %d\n",
            matrix_sizes[0],
            matrix_sizes[0]);
    }
    else {
        printf("Matrix sizes: 500, 1000, 1500\n");
    }

    printf("Max OpenMP threads: %d\n",
        omp_get_max_threads());


    // ========================================================
    // Запуск экспериментов для каждого N
    // ========================================================

    for (int size_id = 0; size_id < num_sizes; size_id++) {

        int N = matrix_sizes[size_id];

        printf("\n");
        printf("==================================================\n");
        printf("N = %d\n", N);
        printf("==================================================\n");


        // ====================================================
        // Выделение памяти
        // ====================================================

        double* A = alloc_matrix(N);
        double* B = alloc_matrix(N);

        double* C_seq = alloc_matrix(N);
        double* C_par = alloc_matrix(N);


        // ====================================================
        // Инициализация
        // ====================================================

        init_matrix(A, N);
        init_matrix(B, N);


        // ====================================================
        // ПРОВЕРКА ПРАВИЛЬНОСТИ
        // ====================================================

        int check_threads =
            std::min(24, omp_get_max_threads());

        omp_set_num_threads(check_threads);

        double check_time;

        blas_dgemm(
            N, N, N,
            A, B, C_par,
            check_time
        );

        dgemm_seq(
            N, N, N,
            A, B, C_seq
        );

        double max_diff = compare_matrices(
            C_seq,
            C_par,
            N
        );

        printf("\nCorrectness check\n");
        printf("-----------------\n");
        printf("Max difference = %.12e\n", max_diff);

        if (max_diff < EPS) {
            printf("CHECK: OK\n");
        }
        else {
            printf("CHECK: FAILED\n");
        }


        // ====================================================
        // SEQUENTIAL — 10 запусков
        // ====================================================

        double seq_time = measure_seq(
            N, N, N,
            A, B, C_seq
        );

        double seq_gflops = calculate_gflops(
            N, N, N,
            seq_time
        );

        printf("\nSequential\n");
        printf("-----------------\n");
        printf("Average time = %.6f sec\n", seq_time);
        printf("GFLOPS       = %.3f\n", seq_gflops);


        // ====================================================
        // PARALLEL — 10 запусков для каждого P
        // ====================================================

        int thread_counts[] = {
            1, 2, 4, 8, 16, 24
        };

        int num_tests =
            sizeof(thread_counts) /
            sizeof(thread_counts[0]);

        printf("\nParallel\n");
        printf("--------------------------------------------------\n");
        printf("Threads       Time (sec)       GFLOPS       Speedup\n");
        printf("--------------------------------------------------\n");

        for (int p = 0; p < num_tests; p++) {

            int threads = thread_counts[p];

            if (threads > omp_get_num_procs()) {
                continue;
            }

            omp_set_num_threads(threads);

            double par_time = measure_parallel(
                N, N, N,
                A, B, C_par
            );

            double par_gflops = calculate_gflops(
                N, N, N,
                par_time
            );

            double speedup =
                seq_time / par_time;

            printf(
                "%7d       %.6f        %.3f        %.3f\n",
                threads,
                par_time,
                par_gflops,
                speedup
            );
        }


        // ====================================================
        // Освобождение памяти
        // ====================================================

        free(A);
        free(B);
        free(C_seq);
        free(C_par);
    }


    return 0;
}