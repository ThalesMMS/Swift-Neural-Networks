=============================================================================
MLP (Multi-Layer Perceptron) - Redes Neurais em C
=============================================================================

AUTORES: Antonio Neto e Thales Matheus

=============================================================================
1. DESCRIÇÃO
=============================================================================

Este projeto implementa Redes Neurais Artificiais (Multi-Layer Perceptron)
em C puro, incluindo dois exemplos:

1. mnist_mlp.c - Classificador de dígitos MNIST
2. mlp_simple.c - Exemplo educacional (problema XOR)

=============================================================================
2. ARQUIVOS DO PROJETO
=============================================================================

CÓDIGO FONTE:
- mnist_mlp.c      : MLP para classificação MNIST (784→512→10)
- mlp_simple.c     : MLP educacional para XOR (2→4→1)

SCRIPTS:
- data_download.sh : Download automático do dataset MNIST
- plot_comparison.py : Gera gráficos de treinamento
- mlp_manim.py     : Animação da arquitetura da rede

CONFIGURAÇÃO:
- Makefile         : Comandos de build
- run.txt          : Referência rápida de comandos
- requirements.txt : Dependências Python (numpy, pillow)

DADOS:
- logs/            : Logs de treinamento

=============================================================================
3. MNIST_MLP - CLASSIFICADOR DE DÍGITOS
=============================================================================

ARQUITETURA:
- Entrada: 784 neurônios (28×28 pixels)
- Oculta: 512 neurônios (ReLU)
- Saída: 10 neurônios (Softmax)

PARÂMETROS:
- Learning Rate: 0.01
- Batch Size: 64
- Épocas: 10

DATASET:
- 60.000 imagens de treino
- 10.000 imagens de teste
- Acurácia esperada: ~97%

=============================================================================
4. MLP_SIMPLE - EXEMPLO EDUCACIONAL
=============================================================================

ARQUITETURA:
- Entrada: 2 neurônios
- Oculta: 4 neurônios (Sigmoid)
- Saída: 1 neurônio (Sigmoid)

PROBLEMA:
- XOR (ou-exclusivo)
- 4 amostras de treino
- 1.000.000 épocas

=============================================================================
5. COMPILAÇÃO E EXECUÇÃO
=============================================================================

PRÉ-REQUISITOS:
- GCC
- Make (opcional)

COMPILAÇÃO:
    gcc -O3 -o mnist_mlp mnist_mlp.c -lm
    gcc -O3 -o mlp_simple mlp_simple.c -lm

Ou usando Make:
    make build_mnist
    make build_simple

EXECUÇÃO:
    ./mnist_mlp
    ./mlp_simple

=============================================================================
6. DOWNLOAD DO DATASET MNIST
=============================================================================

Para baixar automaticamente:
    make data_download

Ou baixe de:
    https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Arquivos necessários em ./data/:
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte

=============================================================================
7. VISUALIZAÇÃO
=============================================================================

Para gerar gráficos de treinamento:
    python plot_comparison.py

=============================================================================
8. REFERÊNCIAS
=============================================================================

Código base adaptado de:
https://github.com/djbyrne/mlp.c

MNIST Database:
http://yann.lecun.com/exdb/mnist/

=============================================================================
