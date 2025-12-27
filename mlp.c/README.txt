=============================================================================
TRABALHO PRÁTICO - PROGRAMAÇÃO PARALELA
MLP (Multi-Layer Perceptron) para Classificação MNIST
=============================================================================

AUTORES: Antonio Neto e Thales Matheus
DATA: Junho 2025

=============================================================================
1. DESCRIÇÃO DA APLICAÇÃO
=============================================================================

Este projeto implementa uma Rede Neural Artificial (Multi-Layer Perceptron - MLP)
para classificação de dígitos manuscritos do dataset MNIST.

ARQUITETURA DA REDE:
- Camada de Entrada: 784 neurônios (imagens 28x28 pixels)
- Camada Oculta: 512 neurônios (ativação ReLU)
- Camada de Saída: 10 neurônios (ativação Softmax, classes 0-9)

DATASET:
- 60.000 imagens de treino
- 10.000 imagens de teste
- Fonte: MNIST Database (http://yann.lecun.com/exdb/mnist/)

ALGORITMO DE TREINAMENTO:
- Stochastic Gradient Descent (SGD)
- Mini-batches de 64 amostras
- Learning Rate: 0.01
- Épocas: 10

CATEGORIA DE IA: Rede Neural (Perceptron Multicamadas)

=============================================================================
2. VERSÕES IMPLEMENTADAS
=============================================================================

Este projeto contém 4 versões do código:

1. VERSÃO SEQUENCIAL (mnist_mlp.c)
   - Implementação base sem paralelização
   - Execução single-thread

2. VERSÃO OPENMP CPU (mnist_mlp_openmp_cpu.c)
   - Paralelização com OpenMP para multicore
   - Utiliza diretivas #pragma omp parallel for
   - Paralelização de: forward pass, backward pass, weight updates

3. VERSÃO OPENMP GPU (mnist_mlp_openmp_gpu.c)
   - Implementação híbrida CPU+GPU
   - GPU: Multiplicação de matriz da camada oculta (784×512)
   - CPU: Camada de saída, backward pass, weight updates
   - Utiliza #pragma omp target para offloading

4. VERSÃO CUDA (mnist_mlp_cuda.cu)
   - Paralelização completa em GPU com CUDA
   - Kernels customizados para forward, backward, weight updates
   - Memória unificada (cudaMallocManaged)

=============================================================================
3. DOWNLOAD DO DATASET
=============================================================================

O dataset MNIST deve estar na pasta ./data/

Para baixar automaticamente:
    make data_download

Ou baixe manualmente de:
    https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Arquivos necessários:
    ./data/train-images.idx3-ubyte
    ./data/train-labels.idx1-ubyte
    ./data/t10k-images.idx3-ubyte
    ./data/t10k-labels.idx1-ubyte

=============================================================================
4. INSTRUÇÕES DE COMPILAÇÃO
=============================================================================

PRÉ-REQUISITOS:
- GCC com suporte a OpenMP
- CUDA Toolkit (para versão CUDA)
- Make (opcional, para usar Makefile)

COMPILAÇÃO MANUAL:

# Versão Sequencial
gcc -O3 -o mnist_mlp mnist_mlp.c -lm

# Versão OpenMP CPU
gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm

# Versão OpenMP GPU (requer GCC com suporte nvptx-none)
gcc -fopenmp -foffload=nvptx-none -O3 -fno-lto -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm
# Nota: Se não tiver suporte GPU, compilar sem offloading funciona em CPU:
gcc -fopenmp -O3 -o mnist_mlp_openmp_gpu mnist_mlp_openmp_gpu.c -lm

# Versão CUDA
nvcc -O3 -o mnist_mlp_cuda mnist_mlp_cuda.cu

=============================================================================
5. INSTRUÇÕES DE EXECUÇÃO
=============================================================================

EXECUÇÃO DIRETA:

# Versão Sequencial
./mnist_mlp

# Versão OpenMP CPU (exemplo com 4 threads)
export OMP_NUM_THREADS=4
./mnist_mlp_openmp_cpu

# Versão OpenMP GPU
./mnist_mlp_openmp_gpu

# Versão CUDA
./mnist_mlp_cuda

EXECUÇÃO COM MAKEFILE:
make run_sequential
make run_openmp_cpu
make run_openmp_gpu
make run_cuda

CONTROLE DE THREADS (OpenMP):
Para testar com diferentes números de threads:
    export OMP_NUM_THREADS=1
    export OMP_NUM_THREADS=2
    export OMP_NUM_THREADS=4
    export OMP_NUM_THREADS=8
    export OMP_NUM_THREADS=16
    export OMP_NUM_THREADS=32

=============================================================================
6. TEMPOS DE EXECUÇÃO (10 ÉPOCAS)
=============================================================================

AMBIENTE DE TESTE:
- CPU: Intel Core i7-15th Gen
- GPU: NVIDIA RTX 3050 Ti
- RAM: 16 GB
- OS: Windows 11

VERSÃO SEQUENCIAL:
- Tempo total: 337.11 segundos
- Tempo/época: 33.71 segundos
- Acurácia: ~97.4%

VERSÃO OPENMP CPU:
- 1 thread:    42.79s/época (Speedup: 0.79×) - 27% mais lento
- 2 threads:   37.73s/época (Speedup: 0.89×) - 12% mais lento
- 4 threads:   35.44s/época (Speedup: 0.95×) - 5% mais lento
- 8 threads:   49.94s/época (Speedup: 0.68×) - 48% mais lento
- 16 threads:  71.44s/época (Speedup: 0.47×) - 112% mais lento
- 32 threads: 102.83s/época (Speedup: 0.33×) - 205% mais lento

VERSÃO OPENMP GPU:
- Tempo total: 1273.21 segundos
- Tempo/época: 127.32 segundos
- Speedup: 0.26× (quase 4× mais lento que sequential!)

VERSÃO CUDA:
- Tempo total: 111.48 segundos
- Tempo/época: 11.15 segundos
- Speedup: 3.02× (MELHOR PERFORMANCE!)

OBSERVAÇÕES:
- Todos os tempos são para 10 épocas completas de treinamento
- Speedup calculado em relação à versão sequencial (33.71s/época)
- OpenMP CPU apresenta NEGATIVE SCALING (piora com mais threads!)
- OpenMP GPU tem overhead muito alto para problema deste tamanho
- CUDA obtém speedup real de 3× devido a paralelismo massivo
- Logs salvos em ./logs/training_loss_*.txt

=============================================================================
7. RESULTADOS ESPERADOS
=============================================================================

Todas as versões devem produzir acurácia similar no dataset de teste:
- Acurácia esperada: ~95-97% (10 épocas)
- Loss final: ~0.05-0.10

ARQUIVOS DE SAÍDA:
- mnist_model.bin         (Versão Sequencial)
- mnist_model_cpu.bin     (OpenMP CPU)
- mnist_model_gpu.bin     (OpenMP GPU)
- mnist_model_cuda.bin    (CUDA)

LOGS DE TREINAMENTO:
- ./logs/training_loss_c.txt
- ./logs/training_loss_openmp_cpu.txt
- ./logs/training_loss_openmp_gpu.txt
- ./logs/training_loss_cuda.txt

=============================================================================
8. VISUALIZAÇÃO DOS RESULTADOS
=============================================================================

Para gerar gráficos comparativos:
    python plot_comparison.py

Isso gera: training_comparison.png com:
- Loss vs Época
- Tempo por Época
- Tempo Cumulativo
- Comparação de Speedup

=============================================================================
9. MUDANÇAS PARA PARALELIZAÇÃO
=============================================================================

9.1 OPENMP CPU (mnist_mlp_openmp_cpu.c):

MUDANÇA 1 - Estrutura de Dados:
    - Arrays 2D (double **) convertidos para 1D (double *)
    - Acesso: weights[i][j] → weights[i * output_size + j]
    - Motivo: Melhor localidade de cache e preparação para GPU

MUDANÇA 2 - Forward Pass (Linhas ~280):
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    Paraleliza cálculo de cada neurônio independentemente

MUDANÇA 3 - Activation Functions (Linhas ~298):
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    Paraleliza aplicação de ReLU/Sigmoid

MUDANÇA 4 - Backward Pass (Linhas ~328, ~341):
    #pragma omp parallel for
    Paraleliza cálculo de deltas e gradientes

MUDANÇA 5 - Weight Updates (Linhas ~370):
    #pragma omp parallel for collapse(2)
    Paraleliza atualização de pesos (dois loops aninhados)

MUDANÇA 6 - Medição de Tempo (Linha ~462):
    clock() → omp_get_wtime()
    Motivo: clock() soma tempo de todas as threads

9.2 OPENMP GPU (mnist_mlp_openmp_gpu.c):

MUDANÇA 1 - GPU Offloading (Linha ~275):
    #pragma omp target teams distribute parallel for
    map(to: inputs[...], weights[...], biases[...])
    map(from: outputs[...])

    Offload da multiplicação de matriz da camada oculta para GPU
    - to: Transfere dados CPU → GPU
    - from: Transfere resultados GPU → CPU
    - teams: Cria blocos de threads (similar a CUDA blocks)
    - distribute: Distribui iterações entre blocos
    - parallel for: Paraleliza dentro de cada bloco

MUDANÇA 2 - Abordagem Híbrida:
    - GPU: Hidden layer forward (784×512 - maior custo computacional)
    - CPU: Output layer, backward pass (operações menores)
    - Motivo: Overhead de kernel launch só vale para operações grandes

9.3 CUDA (mnist_mlp_cuda.cu):

MUDANÇA 1 - Kernels CUDA:
    __global__ void linear_forward_kernel(...)
    __global__ void relu_kernel(...)
    __global__ void update_weights_kernel(...)

    Implementação de kernels customizados para cada operação

MUDANÇA 2 - Gerenciamento de Memória:
    cudaMallocManaged() - Memória unificada CPU/GPU
    Simplifica transferências de dados

MUDANÇA 3 - Lançamento de Kernels:
    int blocks = (size + 255) / 256;
    kernel<<<blocks, 256>>>(...);

    256 threads por bloco (típico para GPUs NVIDIA)
    Número de blocos calculado dinamicamente

MUDANÇA 4 - Sincronização:
    cudaDeviceSynchronize()
    Garante que GPU terminou antes de CPU acessar dados

=============================================================================
10. CÓDIGO FONTE BASE
=============================================================================

A versão sequencial base foi adaptada de:
https://github.com/djbyrne/mlp.c/blob/main/mnist_mlp.c

Modificações incluem:
- Aumento da camada oculta (128 → 512 neurônios)
- Implementação de salvamento/carregamento de modelo
- Logging de métricas de treinamento
- Estruturas de dados otimizadas

=============================================================================
11. DOCUMENTAÇÃO ADICIONAL
=============================================================================

ARQUIVOS DE DOCUMENTAÇÃO:
- IMPLEMENTATION_GUIDE.md: Guia técnico detalhado das implementações
- README.md: Documentação original do projeto base
- plot_comparison.py: Script para geração de gráficos

REFERÊNCIAS:
- OpenMP Specification: https://www.openmp.org/specifications/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- MNIST Database: http://yann.lecun.com/exdb/mnist/

=============================================================================
12. CONTATO E SUPORTE
=============================================================================

Em caso de dúvidas ou problemas:
- GitHub: https://github.com/nietus/Computacao-Paralela
- Email: antonio@deeptera.com

=============================================================================
