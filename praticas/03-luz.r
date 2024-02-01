library(torch)
library(luz)
library(torchvision)

cars_scale <- cars |>
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

# dados em matriz
cars_matrix <- model.matrix(~speed, data = cars_scale)
Xy <- cbind(cars_matrix, cars_scale$dist)

# dados em tensor
cars_tensor <- torch_tensor(Xy)

# dados que vamos usar na rede neural
xx <- cars_tensor[, 2]$unsqueeze(2)
yy <- cars_tensor[, 3]$unsqueeze(2)


# ATÉ agora, vimos isso aqui:

mlp <- nn_sequential(
  nn_linear(1, 8),
  nn_relu(),
  nn_linear(8, 1)
)

learning_rate <- 0.01
optimizer <- optim_sgd(mlp$parameters, lr = learning_rate)

l <- nn_mse_loss(reduction = "mean")

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

library(ggplot2)
ggplot(cars_scale) +
  aes(x = speed, y = dist) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_line(
    colour = "red",
    data = data.frame(speed = cars_scale$speed, dist = as.numeric(y_pred))
  )

# Será que dá para simplificar ainda mais?

## dataset() e dataloader() --------------------------------------------------

# dataset() é um objeto cuja principal finalidade é retornar um item de dados
# e seu rótulo correspondente. Ele recebe um objeto que pode ser indexado
# (como uma lista, um vetor, um data.frame ou um tensor) e uma função que
# transforma os dados em tensores.

# dataloader() é um objeto que recebe um dataset() e retorna um iterador que
# permite acessar os dados em lotes (batches). No mundo real, é muito comum
# que os dados sejam muito grandes para serem processados de uma só vez. Por
# isso, é necessário dividi-los em lotes menores.

# Vamos ver como isso funciona na prática.

## exemplo:
# install.packages("torchdatasets")

ds <- torchvision::mnist_dataset("dados/", download = TRUE)
ds$.getitem(1)
ds[1]

# Primeiro, vamos criar um dataset() a partir dos dados que já temos.
# elementos necessários: initialize, length, .getitem

ds <- dataset(
  name = "cars_dataset",
  initialize = function(da) {
    cars_scale <- da |>
      dplyr::mutate(
        speed = scale(speed),
        dist = scale(dist)
      )
    # dados em matriz
    cars_matrix <- model.matrix(~speed, data = cars_scale)
    Xy <- cbind(cars_matrix, cars_scale$dist)

    # dados em tensor
    cars_tensor <- torch_tensor(Xy)

    # dados que vamos usar na rede neural
    self$x <- cars_tensor[, 2]$unsqueeze(2)
    self$y <- cars_tensor[, 3]$unsqueeze(2)
  },
  .length = function() {
    dim(self$x)[1]
  },
  .getitem = function(idx) {
    list(self$x[idx, ], self$y[idx, ])
  }
)

da <- mtcars

ds_mtcars <- dataset(
  name = "cars_dataset",
  initialize = function(da) {
    mtcars_scale <- da |>
      dplyr::mutate(dplyr::across(
        c(mpg ,disp, hp, drat, wt, qsec),
        scale
      ))
    # dados em matriz
    #mtcars_matrix <- model.matrix(~.-mpg, data = mtcars_scale)
    #Xy <- cbind(mtcars_matrix, mtcars_matrix$mpg)
    mtcars_matrix <- as.matrix(mtcars_scale)

    # dados em tensor
    mtcars_tensor <- torch_tensor(mtcars_matrix)

    # dados que vamos usar na rede neural
    self$x <- mtcars_tensor[, 2:-1]
    self$y <- mtcars_tensor[, 1]$unsqueeze(2)
  },
  .length = function() {
    dim(self$x)[1]
  },
  .getitem = function(idx) {
    list(self$x[idx, ], self$y[idx, ])
  }
)

ds_mtcars <- ds_mtcars(mtcars)
ds_mtcars$.getitem(1)
ds_mtcars[1]

dl_mtcars <- dataloader(ds_mtcars, batch_size = 4, shuffle = TRUE)

dl_mtcars |>
  dataloader_make_iter() |>
  dataloader_next()


# também podemos criar um dataset() a partir de tensores
ds_cars_alternativa <- tensor_dataset(xx, yy)
ds_cars_alternativa[1]

# Agora, vamos criar um dataloader() a partir do dataset() que acabamos de
# criar. O dataloader() recebe o dataset() e o tamanho do lote (batch_size).
# O tamanho do lote é o número de itens que serão retornados a cada iteração.

dl_cars <- dataloader(ds_cars, batch_size = 10, shuffle = TRUE)

length(ds_cars)
length(dl_cars)

dl_cars |>
  dataloader_make_iter() |>
  dataloader_next()

## CHEGOU A HORA DA FELICIDADE ------------------------------------------------

# Com o luz, vamos simplificar significativamente nosso código.

net <- nn_module(
  # função de inicialização, para poder ler hiperparâmetros
  # e fazer outras configurações iniciais
  initialize = function(d_hidden) {
    self$net <- nn_sequential(
      nn_linear(1, d_hidden),
      nn_relu(),
      nn_linear(d_hidden, 1)
    )
  },
  # função de forward, que é o que vai ser executado a cada iteração
  # no treinamento do modelo
  forward = function(x) {
    self$net(x)
  }
)

result <- net |>
  # função de perda e otimizador
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_sgd
  ) |>
  # hiperparâmetros do modelo
  set_hparams(
    d_hidden = 8
  ) |>
  # parâmteros do otimizador
  set_opt_hparams(
    lr = 0.01
  ) |>
  fit(dl_cars, epochs = 100)

y_pred <- predict(result, xx)

ggplot(cars_scale) +
  aes(x = speed, y = dist) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_line(
    colour = "red",
    data = data.frame(
      speed = cars_scale$speed,
      dist = as.numeric(y_pred$to(device = "cpu"))
    )
  )

# Legal! Então precisamos definir o dataset, o dataloader, nosso
# módulo de modelagem, e depois tem o bloco de rodar as coisas, com
# a função de perda, otimizador, hiperparâmetros e hiperparâmetros
# de otimização. E, por fim, a função fit().

# Mas, e se a gente quiser fazer uma validação cruzada? E se a gente
# quiser salvar o modelo? E se a gente quiser fazer um grid search
# para encontrar os melhores hiperparâmetros? E se a gente quiser
# fazer um early stopping? E se a gente quiser fazer um ensemble?

# Tudo isso é possível com o luz.

## CNN -----------------------------------------------------------------------

train_ds <- mnist_dataset(
  "dados/",
  download = FALSE,
  train = TRUE,
  transform = transform_to_tensor
)

valid_ds <- mnist_dataset(
  "dados/",
  download = TRUE,
  train = FALSE,
  transform = transform_to_tensor
)

length(train_ds)
length(valid_ds)

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32)


train_iter <- train_dl$.iter()
iter_next <- train_iter$.next()
x <- iter_next$x
y <- iter_next$y

plot(as.raster(as.matrix(x[1,1,,])))

dim(as.matrix(x[1,1,,]))
y[1]

net <- nn_module(

  "MNIST-CNN",

  initialize = function() {
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },

  forward = function(x) {
    x |>
      self$conv1() |>
      nnf_relu() |>
      self$conv2() |>
      nnf_relu() |>
      nnf_max_pool2d(2) |>
      torch_flatten(start_dim = 2) |>
      self$fc1() |>
      nnf_relu() |>
      self$fc2()
  }
)

#?nn_soft_margin_loss
#nn_soft_margin_loss()
#?nn_cross_entropy_loss()
#?nn_nll_loss

fitted <- net |>
  luz::setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz::luz_metric_accuracy()
    )
  ) |>
  luz::fit(
    train_dl,
    epochs = 2,
    valid_data = valid_dl
  )

preds <- predict(fitted, valid_dl)
preds$shape

predict(fitted, valid_ds[1]$x$unsqueeze(1)) |>
  as.numeric() |>
  which.max()

plot(as.raster(as.matrix(valid_ds[1]$x$unsqueeze(1)[1,1,,])))
