@echo off
REM Windows batch script to test OpenMP CPU with different thread counts

echo ===================================================================
echo Testing OpenMP CPU with different thread counts
echo ===================================================================

REM Check if executable exists
if not exist "mnist_mlp_openmp_cpu.exe" (
    echo Error: mnist_mlp_openmp_cpu.exe not found. Please compile first:
    echo   gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Test with 1 thread
echo.
echo -------------------------------------------------------------------
echo Running with 1 thread...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=1
mnist_mlp_openmp_cpu.exe

REM Test with 2 threads
echo.
echo -------------------------------------------------------------------
echo Running with 2 threads...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=2
mnist_mlp_openmp_cpu.exe

REM Test with 4 threads
echo.
echo -------------------------------------------------------------------
echo Running with 4 threads...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=4
mnist_mlp_openmp_cpu.exe

REM Test with 8 threads
echo.
echo -------------------------------------------------------------------
echo Running with 8 threads...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=8
mnist_mlp_openmp_cpu.exe

REM Test with 16 threads
echo.
echo -------------------------------------------------------------------
echo Running with 16 threads...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=16
mnist_mlp_openmp_cpu.exe

REM Test with 32 threads
echo.
echo -------------------------------------------------------------------
echo Running with 32 threads...
echo -------------------------------------------------------------------
set OMP_NUM_THREADS=32
mnist_mlp_openmp_cpu.exe

echo.
echo ===================================================================
echo All tests complete!
echo ===================================================================
echo.
echo Log files created in logs/ directory:
echo   - training_loss_openmp_cpu_1threads.txt
echo   - training_loss_openmp_cpu_2threads.txt
echo   - training_loss_openmp_cpu_4threads.txt
echo   - training_loss_openmp_cpu_8threads.txt
echo   - training_loss_openmp_cpu_16threads.txt
echo   - training_loss_openmp_cpu_32threads.txt
