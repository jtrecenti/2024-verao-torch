library(torch)
library(zeallot)

# Na aula anterior, vimos como criar tensores e realizar operações
# Acabamos a aula com um exemplo de broadcast, que é uma técnica que
# permite realizar operações aritméticas em tensores de diferentes formas
# Nesta aula, vamos retomar o assunto de broadcast e veremos como realizar
# operações matriciais e decomposições de matrizes com o pacote torch.

# Depois, vamos ver os conceitos de autograd e gradientes, que são
# fundamentais para o treinamento de redes neurais.
# Por último, veremos os módulos, que são a base para a construção de
# redes neurais com o pacote torch.

# Broadcasting ---------------------------------------------------------------

# Broadcasting é uma técnica que permite realizar operações aritméticas
# em tensores de diferentes formas

# Criando dois tensores de formas diferentes
(t_a <- torch_rand(c(3, 1)))
(t_b <- torch_rand(c(1, 4)))

c(3, 4)
c(3, 4)

(t_a <- torch_arange(1,3)$view(c(3, 1)))

(t_b <- torch_arange(1,4)$view(c(1, 4)))

t_a + t_b

a <- torch_tensor(matrix(c(c(1,2,3), c(1,2,3), c(1,2,3), c(1,2,3)), ncol = 4))

b <- torch_tensor(matrix(c(c(1,2,3,4), c(1,2,3,4), c(1,2,3,4)), ncol = 4, byrow = TRUE))

a+b


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

t1$matmul(t2)

t1$mm(t2)

# Transposta

# Transposta com torch_t()
(t6 <- torch_t(t1))

# ou então
(t7 <- t1$t())

# t1$t_(): inplace

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

## vamos usar o dataset cars para ilustrar as operações matriciais
## e decomposições. É bom calcular as escalas antes de começar
## a trabalhar com os dados, para evitar problemas numéricos

cars_scale <- cars |>
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

cars_matrix <- model.matrix(~speed, data = cars_scale)
Xy <- cbind(cars_matrix, cars_scale$dist)
cars_tensor <- torch_tensor(Xy)

(X <- cars_tensor[, 1:-2])
(y <- cars_tensor[, -1])

dim(X)
dim(y)

# Solução de regressão linear (mais na próxima aula)
linalg_lstsq(X, y)$solution

coef(lm(speed ~ dist, data = cars_scale))

# Solução da regressão linear "na mão" (X'X)^-1 X'y

X$t()$mm(X)

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

#torch_norm(Q[,1])
#torch_dot(Q[,1], Q[,2])

# forma alternativa com pacote zeallot:
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

# Autograd e gradientes ------------------------------------------------------

# Autograd é uma técnica que permite calcular gradientes automaticamente
# para funções que envolvem tensores. É a base para o treinamento de redes
# neurais.

# Para calcular gradientes, precisamos criar tensores com o parâmetro
# requires_grad = TRUE

# Criando um tensor com requires_grad = TRUE
# o valor aqui é arbitrário!
(t1 <- torch_tensor(10, requires_grad = TRUE))

# agora vamos fazer algumas contas com ele
(t2 <- t1 + 2)
(t3 <- t2$square())
(t4 <- t3 * 3)

# agora, vamos calcular o gradiente de t4 em relação a t1
t4$backward()

t1$grad

# vamos calcular na mão:
# t4 = 3 * (t1 + 2)^2
# dt4/dt1 = 3 * 2 * (t1 + 2) = 6 * (t1 + 2)
6 * (t1 + 2)

# o que acontece se quisermos acessar o gradiente dos passos intermediários?
t2$grad
t3$grad

# resolvendo com retain_grad()
t1 <- torch_tensor(10, requires_grad = TRUE)
t2 <- t1 + 2
t2$retain_grad()
t3 <- t2$square()
t3$retain_grad()
t4 <- t3 * 3
t4$retain_grad()

t4$backward()

t3$grad # dt4/dt3 = 3

t2$grad # dt4/dt2 = dt4/dt3 * dt3/dt2 = 3 * (2 * t2) = 6 * (t1 + 2)

# dt2/dt1 = 1
t1$grad # dt4/dt1 = dt4/dt2 * dt2/dt1 = 6 * (t1 + 2) * 1 = 6 * (t1 + 2)

# Regressão linear com descida de gradiente -----------------------------------

# Vamos usar o dataset cars para ilustrar a descida de gradiente

# meta código:

# 1. inicializar os parâmetros do modelo
# 2. calcular a função de perda
# 3. calcular a derivada da função de perda em relação aos parâmetros
# 4. atualizar os parâmetros do modelo

# ESSE CÓDIGO NÃO FUNCIONA! É APENAS UM EXEMPLO!
for (i in seq_len(N)) {
  perda <- mse(modelo, beta)
  perda$backward()
  # atualiza os parâmetros
  beta$sub_(lr * beta$grad)
}

# com o torch, precisaremos ter alguns cuidados nesse processo.

## parâmetros da otimização
num_iterations <- 1000
lr <- 0.01 # learning rate, alpha

## parâmetros do modelo
beta <- torch_tensor(c(0, 1), requires_grad = TRUE)

# como calcular o MSE?
resid <- X$matmul(beta) - y
loss <- resid$square()$mean()

## modelo. Isso ficará mais complexo no futuro...
modelo <- function(X, beta) {
  X$matmul(beta)
}

## função de perda
mse <- function(beta) {
  resid <- modelo(X, beta) - y
  loss <- resid$square()$mean()
  loss
}

#1:num_iterations
for (i in seq_len(num_iterations)) {

  if (i %% 100 == 0) cat("Iteração: ", i, "\n")

  perda <- mse(beta)

  if (i %% 100 == 0) {
    cat("Valor da perda: ", as.numeric(perda), "\n")
  }

  # calcula a derivada
  perda$backward()
  if (i %% 100 == 0) {
    cat("A derivada é: ", as.matrix(beta$grad), "\n\n")
  }

  with_no_grad({
    beta$sub_(lr * beta$grad)
    beta$grad$zero_()
  })
}

beta

## GRANDE PARENTESES: HESSIANA

# No torch para R, ainda não temos uma versão nativa de Hessiana.

# Ver esse post aqui:
# https://rgiordan.github.io/code/2022/04/01/rtorch_example.html

# No entanto, existem otimizadores chamados "quasi-newton", que fazem
# o cálculo do Hessiano de forma aproximada, usando apenas o gradiente.
# O algoritmo mais famoso nesse sentido é chamado de
# L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).

# Ver esse post aqui:
# https://blogs.rstudio.com/ai/posts/2021-04-22-torch-for-optimization/


num_iterations <- 4
beta <- torch_tensor(c(0, 1), requires_grad = TRUE)

# esse é nosso otimizador!
# esse conceito será bem útil quando começarmos a trabalhar com redes neurais
optimizer <- optim_lbfgs(beta)

atualiza_parametros_com_perda <- function() {

  # veja que não precisamos mais atualizar os parâmetros na mão
  optimizer$zero_grad()

  perda <- mse(beta)
  cat("Perda: ", as.numeric(perda), "\n")

  perda$backward()
  cat("Gradiente: ", as.matrix(beta$grad), "\n\n")

  perda

}

# isso é um módulo!
optimizer

for (i in seq_len(num_iterations)) {
  cat("Iteration: ", i, "\n")
  optimizer$step(atualiza_parametros_com_perda)
}

beta

# Multi Layer Perceptron =====================================================
library(torch)

# zz <- torch_tensor(-1, requires_grad = TRUE)
# zzz <- zz$abs()
# zzz$backward()
# zz$grad

cars_scale <- cars |>
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

cars_scale |>
  ggplot2::ggplot(ggplot2::aes(x = speed, y = dist)) +
  ggplot2::geom_point()

cars_matrix <- model.matrix(~speed, data = cars_scale)
Xy <- cbind(cars_matrix, cars_scale$dist)
cars_tensor <- torch_tensor(Xy)

X <- cars_tensor[, 1:-2]
y <- cars_tensor[, -1]

# Rede neural com descida de gradiente ---------------------------------------

# Agora, vamos fazer uma rede neural na mão!

# primeiro, separamos os parâmetros do intercepto (que vamos chamar de bias, b)
# e os parâmetros das variáveis explicativas (que vamos chamar de pesos, w)

b <- torch_zeros(1, 1, requires_grad = TRUE)
w <- torch_randn(1, 1, requires_grad = TRUE)

# agora, o X não precisa mais daquela coluna com 1's
xx <- X[, -1]$unsqueeze(2)

# o y precisa ter a dimensão arrumada também
yy <- y$unsqueeze(2)

xx$matmul(w) + b
# ou
xx$mm(w)$add(b)

# até aqui, temos exatamente a mesma regressão linear que fizemos antes

# hidden layers

# aqui a dimensão 1 é a dimensão dos dados (que só tem 1 covariável), e
# o 8 é a dimensão de saída do passo atual, que é arbitrária
# (é um hiperparâmetro)
w1 <- torch_randn(1, 8, requires_grad = TRUE)
b1 <- torch_zeros(1, 8, requires_grad = TRUE)

# agora precisamos dos pesos para voltar à dimensão do y!

w2 <- torch_randn(8, 1, requires_grad = TRUE)
b2 <- torch_randn(1, 1, requires_grad = TRUE)

# em geral:

# dimensionalidade do input (1, no nosso caso)
d_in <- 1
# dimensionalidade do hidden layer
d_hidden <- 8
# dimensionalidade do output (1)
d_out <- 1

# pesos que levam do input ao hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# pesos que levam do hidden layer ao output
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# intercepto do hidden layer
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# intercepto do output
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

# como as duas operações são lineares, adicionamos uma camada não linear
# na conta para que a rede neural possa aprender relações não lineares
xx$mm(w)$add(b)$relu()

relu <- function(x) {
  max(0, x)
}

relu(-1)
learning_rate <- 0.01

for (t in 1:1000) {

  ### MODELO
  resultado_1 <- xx$mm(w1)$add(b1)
  nao_linear <- resultado_1$relu()
  y_pred <- nao_linear$mm(w2)$add(b2)
  ## obs: tudo na mesma linha:
  ## y_pred <- xx$mm(w1)$add(b1)$relu()$mm(w2)$add(b2)

  ### PERDA
  loss <- (y_pred - yy)$pow(2)$mean()
  if (t %% 10 == 0)
    cat("Iteração (Época): ", t, "   Perda: ", loss$item(), "\n")

  ### GRADIENTE

  loss$backward()

  ### ATUALIZAÇÃO DOS PARÂMETROS

  with_no_grad({
    # atualizar parâmetros
    w1$sub_(learning_rate * w1$grad)
    w2$sub_(learning_rate * w2$grad)
    b1$sub_(learning_rate * b1$grad)
    b2$sub_(learning_rate * b2$grad)

    # zerar gradientes
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()
  })

}

w1
loss

ggplot2::ggplot(cars_scale) +
  ggplot2::aes(x = speed, y = dist) +
  ggplot2::geom_point() +
  ggplot2::geom_smooth(method = "lm", se = FALSE) +
  ggplot2::geom_line(
    colour = "red",
    data = data.frame(speed = cars_scale$speed, dist = as.numeric(y_pred))
  )

# Módulos --------------------------------------------------------------------

# Módulos são a base para a construção de redes neurais com o pacote torch.
# Eles são objetos que contém parâmetros e métodos para calcular a saída
# de uma rede neural.

# Vamos criar um módulo para uma rede neural com uma camada linear e uma
# função de ativação ReLU.

mlp <- nn_sequential(
  nn_linear(1, 8),
  nn_relu(),
  nn_linear(8, 1)
)

# Obs: fazendo um módulo do zero
meu_nn_linear <- nn_module(
  initialize = function(in_features, out_features) {
    self$w <- nn_parameter(torch_randn(
      in_features, out_features
    ))
    self$b <- nn_parameter(torch_zeros(out_features))
  },
  forward = function(input) {
    input$mm(self$w) + self$b
  }
)

mlp$parameters


for (t in 1:1000) {

  ### MODELO
  y_pred <- mlp(xx)

  ### PERDA
  loss <- (y_pred - yy)$pow(2)$mean()
  if (t %% 10 == 0)
    cat("Iteração (Época): ", t, "   Perda: ", loss$item(), "\n")

  ### GRADIENTE

  loss$backward()

  ### ATUALIZAÇÃO DOS PARÂMETROS

  with_no_grad({

    # atualizar parâmetros
    mlp$parameters[[1]]$sub_(learning_rate * mlp$parameters[[1]]$grad)
    mlp$parameters[[2]]$sub_(learning_rate * mlp$parameters[[2]]$grad)
    mlp$parameters[[3]]$sub_(learning_rate * mlp$parameters[[3]]$grad)
    mlp$parameters[[4]]$sub_(learning_rate * mlp$parameters[[4]]$grad)

    # zerar gradientes
    mlp$parameters[[1]]$grad$zero_()
    mlp$parameters[[2]]$grad$zero_()
    mlp$parameters[[3]]$grad$zero_()
    mlp$parameters[[4]]$grad$zero_()
  })

}

loss

# Otimizadores --------------------------------------------------------------

# Otimizadores são objetos que contém métodos para atualizar os parâmetros

mlp <- nn_sequential(
  nn_linear(1, 8),
  nn_relu(),
  nn_linear(8, 1)
)

optimizer <- optim_sgd(mlp$parameters, lr = learning_rate)

for (t in 1:1000) {

  ### MODELO
  y_pred <- mlp(xx)

  ### PERDA
  loss <- (y_pred - yy)$pow(2)$mean()
  if (t %% 10 == 0)
    cat("Iteração (Época): ", t, "   Perda: ", loss$item(), "\n")

  ### GRADIENTE

  # precisa zerar os gradientes!!!
  optimizer$zero_grad()
  loss$backward()

  ### ATUALIZAÇÃO DOS PARÂMETROS
  # bem mais simples!
  optimizer$step()

}

loss

# Função de perda -----------------------------------------------------------

mlp <- nn_sequential(
  nn_linear(1, 8),
  nn_relu(),
  nn_linear(8, 1)
)

optimizer <- optim_sgd(mlp$parameters, lr = learning_rate)

l <- nn_mse_loss(reduction = "mean")

# nnf_mse_loss(1, 1)


for (t in 1:1000) {

  ### MODELO
  y_pred <- mlp(xx)

  ### PERDA
  loss <- l(y_pred, yy)
  if (t %% 10 == 0)
    cat("Iteração (Época): ", t, "   Perda: ", loss$item(), "\n")

  ### GRADIENTE
  optimizer$zero_grad()
  loss$backward()

  ### ATUALIZAÇÃO DOS PARÂMETROS
  optimizer$step()

}

loss
