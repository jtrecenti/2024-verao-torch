library(torch)

# O que é um tensor? ---------------------------------------------------------

# Em frameworks de deep learning como TensorFlow e PyTorch, tensores são
# arrays multidimensionais otimizados para computação rápida.

# Criando um tensor simples

t1 <- torch_tensor(1)
print(t1)

# Propriedades de um tensor
# Podemos verificar o tipo de dados, o dispositivo e a forma do tensor
t1$dtype
t1$device
t1$shape

# Obs: vamos utilizar a sintaxe do () para imprimir os objetos
# ao mesmo tempo da atribuição. Exemplo:

(t1 <- torch_tensor(1))

# Mudando propriedades de um tensor
# Alterando o tipo de dados e o dispositivo (GPU, se disponível)

t2 <- t1$to(dtype = torch_int())

t2$dtype

# t2 <- t1$to(device = "cuda") # Descomente se tiver GPU
# print(t2$device)

# Alterando a forma do tensor
(t3 <- t1$view(c(1, 1)))

t3$shape

# Criando tensores -----------------------------------------------------------
# Há várias maneiras de criar tensores, incluindo a partir de valores,
# especificações ou datasets

# Tensores a partir de valores
(t4 <- torch_tensor(1:5))

# Tensores a partir de especificações
# Por exemplo, criando um tensor 3x3 com valores normalmente distribuídos
(t5 <- torch_randn(3, 3))

(t5 <- torch_rand(3, 3))

# Tensores a partir de datasets

# Supondo que temos um dataset em um dataframe chamado 'meu_dataframe'

cars
(meu_tensor <- torch_tensor(as.matrix(cars)))

# Note que é necessário transformar o dataframe em uma matriz para que o
# torch_tensor funcione. Isso pode dar mais trabalho em casos envolvendo
# datasets mais complexos, como dados de textos.

# Operações em tensores ------------------------------------------------------
# Podemos realizar operações matemáticas comuns com tensores

t6 <- torch_tensor(c(1, 2))
t7 <- torch_tensor(c(3, 4))

# Adição
(resultado <- torch_add(t6, t7))

t6 + t7 # Equivalente a torch_add(t6, t7)

# Adição in-place (modifica o tensor original). Cuidado!!!
t6$add_(t7)
t6

# Operações de matriz, como produto escalar
(t8 <- t6$dot(t7))

# Acessando partes de um tensor (Slicing e Indexação) ------------------------

# Considerando um tensor 3D para exemplo
(t <- torch_tensor(array(1:27, dim = c(3, 3, 3))))

# Slicing: Selecionando um subconjunto do tensor
# Exemplo: selecionando a primeira "página" do tensor 3D

(primeira_pagina <- t[1, ..])

# Indexação: Acessando um elemento específico
# Exemplo: acessando o elemento na primeira linha, segunda coluna da
# primeira página
(elemento <- t[1, 2, ..])

# Combinações de slicing e indexação
# Exemplo: selecionando a primeira e a terceira coluna da segunda página
(colunas_1_3_pag_2 <- t[2, .., c(1, 3)])
# (colunas_1_3_pag_2 <- t[2, , c(1, 3)])

# No torch, o -1 funciona para pegar o último elemento
# Exemplo: selecionando a última coluna da última página
t[3, ..]
(ultima_coluna_ultima_pagina <- t[3, .., -1])


# Redimensionando tensores ---------------------------------------------------

# Inicializando um tensor 1D
(t_flat <- torch_arange(1, 12))

# Redimensionando para uma matriz 3x4
(t_matrix <- t_flat$view(c(3, 4)))

# Redimensionando para um tensor 3D 2x2x3
(t_3d <- t_flat$view(c(2, 2, 3)))

# Verificando se os dados são compartilhados entre as formas
print(t_flat$storage()$data_ptr() == t_matrix$storage()$data_ptr())

# Squeeze e Unzqueeze

# Squeeze: remove dimensões com tamanho 1

# Criando um tensor 3D com uma dimensão de tamanho 1

(t_3d <- torch_randn(c(1, 2, 1)))

# Removendo a dimensão de tamanho 1
t_3d$squeeze()

# Unsqueeze: adiciona dimensões com tamanho 1
# Criando um tensor 2D
(t_2d <- torch_randn(c(2, 2)))

# Adicionando uma dimensão de tamanho 1
t_2d$unsqueeze(1)

# O pacote torch fornece dois métodos para mudar a forma de um tensor: view()
# e reshape(). Ambos parecem fazer a mesma coisa, mas há diferenças
# importantes na maneira como operam.

# view():
# - O método view() é usado para redimensionar um tensor sem copiar os dados.
# - Ele retorna uma nova visão do tensor original com a forma especificada.
# - É importante que o novo formato seja compatível com o tamanho original do tensor.
# - view() requer que o tensor original seja contíguo na memória.
#   Se o tensor não for contíguo, view() pode falhar.

# reshape():
# - O método reshape() também é usado para mudar a forma de um tensor.
# - Diferente de view(), reshape() não exige que o tensor original seja contíguo na memória.
# - Se possível, reshape() retornará uma nova visão do tensor original sem copiar os dados.
# - Se o tensor não for contíguo, reshape() criará uma cópia dos dados com a nova forma.

# Demonstração prática:

# Criando um tensor exemplo
(tensor_original <- torch_randn(c(2, 3)))

# Usando view para redimensionar o tensor
# Isso só funcionará se o tensor for contíguo na memória
(tensor_view <- tensor_original$view(c(3, 2)))

# Usando reshape para redimensionar o tensor
# Isso funcionará independentemente de o tensor ser contíguo na memória
(tensor_reshape <- tensor_original$reshape(c(3, 2)))

# A principal diferença entre view e reshape é como eles lidam com tensores não contíguos na memória.
# Enquanto view pode falhar ou exibir comportamento inesperado, reshape garante que a nova forma seja sempre aplicada corretamente,
# criando uma cópia dos dados quando necessário.

# exemplo onde o view falha:

# Criando um tensor não contíguo
(tensor_nao_contiguo <- torch_randn(c(3, 3))$t())

# Usando view para redimensionar o tensor
# Isso falhará porque o tensor não é contíguo na memória

tensor_nao_contiguo$view(c(9))
tensor_nao_contiguo$reshape(c(9))


# Broadcasting ---------------------------------------------------------------

# Broadcasting é uma técnica que permite realizar operações aritméticas
# em tensores de diferentes formas

# Criando dois tensores de formas diferentes
(t_a <- torch_rand(c(3, 1)))
(t_b <- torch_rand(c(1, 4)))

# Broadcasting com escalar
# Multiplicando um tensor por um escalar
t_escalar <- torch_randn(c(3, 3))
(resultado_escalar <- t_escalar * 5)

# Exemplo de broadcasting: adição
# t_a tem forma (3, 1) e t_b tem forma (1, 4)
# O broadcasting expande ambos para a forma (3, 4) e realiza a adição
(resultado_broadcast <- t_a + t_b)

# Exemplo: multiplicando cada linha de t_a pelo vetor t_b
(resultado_multiplicacao <- t_a * t_b)

# O broadcasting permite realizar operações aritméticas entre tensores de
# diferentes formas. As regras do broadcasting são as seguintes:

# Regra 1: Alinhar as formas dos tensores pelo lado direito
# Exemplo: Se temos um tensor A de forma (5, 4) e um tensor B de forma (4,),
#          B é tratado como se tivesse forma (1, 4) para alinhamento.

# Regra 2: Expandir as dimensões onde os tensores têm tamanho 1
# Exemplo: Continuando o exemplo acima, B é expandido para a forma (5, 4),
#          repetindo seus elementos ao longo da primeira dimensão que antes
#          tinha tamanho 1.

# Regra 3: Um tensor pode ser expandido apenas se uma de suas dimensões for 1
# Exemplo: Um tensor de forma (5, 4) pode ser broadcasted com um tensor de
#          forma (1, 4), mas não com um tensor de forma (2, 4), pois 2
#          não é igual a 1 nem a 5.

# Demonstração prática do broadcasting no R com o pacote torch:

# Tensor A com forma (5, 4)
tensor_a <- torch_ones(c(5, 4))

# Tensor B com forma (4,)
tensor_b <- torch_arange(start = 1, end = 4)

# O broadcasting permite somar esses dois tensores,
# embora eles tenham formas diferentes
(resultado <- tensor_a + tensor_b)

# Outro exemplo: tensor C com forma (5, 1)
(tensor_c <- torch_arange(start = 1, end = 5)$unsqueeze(1)$t())

# Somando tensor_a e tensor_c utilizando broadcasting
(resultado2 <- tensor_a + tensor_c)

# O broadcasting é extremamente útil em operações matriciais e de
# manipulação de dados, permitindo evitar loops explícitos e tornando o
# código mais eficiente e legível.

# Operações matriciais -------------------------------------------------------

# Criando dois tensores 2D

(t1 <- torch_randn(c(2, 3)))

(t2 <- torch_randn(c(3, 2)))

# Multiplicação matricial
# Multiplicação matricial com o operador %*% não funciona!
(t3 <- t1 %*% t2)

# Multiplicação matricial com torch_matmul()
(t4 <- torch_matmul(t1, t2))

# Transposta

# Transposta com torch_t()
(t6 <- torch_t(t1))

# ou então
(t7 <- t1$t())

# Determinante

# Determinante com torch_det()
matriz_quadrada <- torch_randn(c(3, 3))
(t9 <- torch_det(matriz_quadrada))

# Inversa
# Inversa com solve() funciona funciona! Mas ele transforma em matriz do R
(t10 <- solve(matriz_quadrada))

# Inversa com torch_inverse() ou linalg_inv() funciona!
(t11 <- torch_inverse(matriz_quadrada))
(t11 <- linalg_inv(matriz_quadrada))

# Decomposições (avançado) ---------------------------------------------------

mtcars_matrix <- as.matrix(scale(mtcars))
mtcars_tensor <- torch_tensor(mtcars_matrix)
X <- mtcars_tensor[, 2:-1]
y <- mtcars_tensor[, 1]

dim(X)
dim(y)

# Solução de regressão linear (mais na próxima aula)
linalg_lstsq(X, y)$solution

# Solução da regressão linear "na mão" (X'X)^-1 X'y

XtX <- torch_matmul(X$t(), X)
Xty <- torch_matmul(X$t(), y)
inv <- linalg_inv(XtX)
inv$matmul(Xty)

# em uma operação só:
torch_matmul(X$t(), X)$inverse() |>
  torch_matmul(X$t()) |>
  torch_matmul(y)

# Decomposição de Cholesky

# Decomposição de Cholesky com linalg_cholesky()
# XtX = L Lt
L <- linalg_cholesky(XtX)

# verificando
LLt <- L$matmul(L$t())
diff <- LLt - XtX
linalg_norm(diff)

# resolvendo regressão
torch_triangular_solve(Xty$unsqueeze(2), L, upper = FALSE) |>
  purrr::pluck(1) |>
  torch_triangular_solve(L$t()) |>
  purrr::pluck(1)

# Decomposição QR

(list_qr <- linalg_qr(X))
(Q <- list_qr[[1]])
(R <- list_qr[[2]])

# forma alternativa com pacote zeallot:
library(zeallot)
c(Q, R) %<-% linalg_qr(X)

# resolvendo
(Qty <- Q$t()$matmul(y))
torch_triangular_solve(Qty$unsqueeze(2), R)[[1]]

# Decomposição SVD

# X = U S Vt
c(U, S, Vt) %<-% linalg_svd(X, full_matrices = FALSE)
U
S
Vt

(Uty <- U$t()$matmul(y))
(y_norm <- Uty / S)
Vt$t()$matmul(y_norm)

