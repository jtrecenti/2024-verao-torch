# Embeddings ----------------------------------------------------------------

library(torch)

# an Embedding module containing 10 tensors of size 3
embedding <- nn_embedding(
  num_embeddings = 10,
  embedding_dim = 3
)
# a batch of 2 samples of 4 indices each
input <- torch_tensor(
  rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)),
  dtype = torch_long()
)
embedding(input)

# example with padding_idx
embedding <- nn_embedding(10, 3, padding_idx = 1)
input <- torch_tensor(
  matrix(c(1, 3, 1, 6), nrow = 1),
  dtype = torch_long()
)
embedding(input)


# Frases aleatórias
frases <- c(
  "eu gosto de gatos",
  "eu gosto de cachorros",
  "eu gosto de gatos e cachorros"
)

# Construindo um dicionário
tokenize <- function(sentences) {
  unlist(strsplit(sentences, " "))
}

tokens <- unique(tokenize(frases))
vocab <- purrr::set_names(seq_along(tokens), tokens)

# Vetorizando as sentenças
vectorized_sentences <- purrr::map(
  frases,
  \(x) as.integer(factor(tokenize(x), levels = tokens))
)

# Já podemos transformar em um objeto do torch
vectorized_sentences <- purrr::map(
  vectorized_sentences,
  torch::torch_tensor
)

# Agora precisamos empilhar eles, mas ainda não dá!

# Maior sequência
max_length <- max(purrr::map_int(vectorized_sentences, length))

# Padding nas sequências
pad_sequences <- purrr::map(
  vectorized_sentences,
  \(x) torch::nnf_pad(x, c(0, max_length - length(x)), value = 7)
)

pad_sequences

# Agora sim!
input_sequences <- torch::torch_stack(pad_sequences)


# Modelo simples com embedding
model <- nn_module(
  "ExampleModel",
  initialize = function(vocab_size, embedding_dim) {
    self$embedding <- nn_embedding(
      num_embeddings = vocab_size + 1,
      embedding_dim = embedding_dim,
      padding_idx = 7
    )
  },
  forward = function(x) {
    embedded <- self$embedding(x)
    embedded
  }
)

# Inicializando o modelo
vocab_size <- length(vocab)
embedding_dim <- 2 # Dimensionality of the embedding vector
net <- model(vocab_size, embedding_dim)


# Forward pass to get embeddings
output <- net(input_sequences)

dim(output)
output[1,..]

## Vamos fazer o mesmo usando o pacote {tok} ------------------------------

## ESSE PACOTE AINDA NÃO ESTÁ COMPLETO E DOCUMENTADO

tokenizer <- tok::tokenizer

tok <- tokenizer$from_pretrained("bert-base-uncased")
tok$encode(frases[1])$ids
tok$encode(frases[2])$ids
tok$encode(frases[3])$ids

x <- tok::encoding

vectorized_sentences <- purrr::map(
  frases,
  \(x) tok$encode(x)$ids
)

# Maior sequência
max_length <- max(purrr::map_int(vectorized_sentences, length))

# Padding nas sequências
pad_sequences <- purrr::map(
  vectorized_sentences,
  \(x) torch::nnf_pad(x, c(0, max_length - length(x)), value = 1)
)

input_sequences <- torch::torch_stack(pad_sequences)

# Inicializando o modelo
vocab_size <- 30000 # https://huggingface.co/bert-base-uncased#preprocessing
embedding_dim <- 2 # Dimensionality of the embedding vector
net <- model(vocab_size, embedding_dim)

# Modelo simples com embedding
model <- nn_module(
  "ExampleModel",
  initialize = function(vocab_size, embedding_dim) {
    self$embedding <- nn_embedding(
      num_embeddings = vocab_size,
      embedding_dim = embedding_dim,
      padding_idx = 1
    )
  },
  forward = function(x) {
    embedded <- self$embedding(x)
    embedded
  }
)

res <- net(input_sequences)

res[3, ..]

# RNNs ---------------------------------------------------------------------

rnn <- nn_rnn(
  input_size = 1,
  hidden_size = 3,
  batch_first = TRUE,
  num_layers = 1
)

output <- rnn(torch_randn(2, 4, 1))

output
dim(output[[1]]) # batch_size, timesteps, hidden_size
dim(output[[2]]) # num_layers, batch_size, hidden_size

# GRU e LSTM ---------------------------------------------------------------

gru <- nn_gru(
  input_size = 1,
  hidden_size = 3,
  batch_first = TRUE,
  num_layers = 1
)

output_gru <- gru(torch_randn(2, 4, 1))

output
dim(output[[1]]) # batch_size, timesteps, hidden_size
dim(output[[2]]) # num_layers, batch_size, hidden_size


lstm <- nn_lstm(
  input_size = 1,
  hidden_size = 3,
  batch_first = TRUE
)

output_lstm <- lstm(torch_randn(2, 4, 1))

output_lstm

# output
dim(output_lstm[[1]]) # batch_size, timesteps, hidden_size

# last hidden state (per layer)
dim(output_lstm[[2]][[1]]) # num_layers, batch_size, hidden_size

# last cell state (per layer)
dim(output_lstm[[2]][[2]]) # num_layers, batch_size, hidden_size

# Exemplo séries temporais ------------------------------------------------

demand_dataset <- dataset(
  name = "demand_dataset",
  initialize = function(x, n_timesteps, sample_frac = 1) {
    self$n_timesteps <- n_timesteps
    self$x <- torch_tensor((x - train_mean) / train_sd)

    n <- length(self$x) - self$n_timesteps

    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1

    list(
      x = self$x[start:end],
      y = self$x[end + 1]
    )
  },
  .length = function() {
    length(self$starts)
  }
)

library(tsibble)
vic_elec <- tsibbledata::vic_elec
vic_elec

demand_hourly <- vic_elec |>
  tsibble::index_by(Hour = lubridate::floor_date(Time, "hour")) |>
  dplyr::summarise(
    Demand = sum(Demand)
  )

demand_train <- demand_hourly |>
  dplyr::filter(lubridate::year(Hour) == 2012) |>
  dplyr::as_tibble() |>
  dplyr::select(Demand) |>
  as.matrix()

demand_valid <- demand_hourly |>
  dplyr::filter(lubridate::year(Hour) == 2013) |>
  dplyr::as_tibble() |>
  dplyr::select(Demand) |>
  as.matrix()

demand_test <- demand_hourly |>
  dplyr::filter(lubridate::year(Hour) == 2014) |>
  dplyr::as_tibble() |>
  dplyr::select(Demand) |>
  as.matrix()


train_mean <- mean(demand_train)
train_sd <- sd(demand_train)

n_timesteps <- 7 * 24

train_ds <- demand_dataset(demand_train, n_timesteps)
valid_ds <- demand_dataset(demand_valid, n_timesteps)
test_ds <- demand_dataset(demand_test, n_timesteps)

train_ds$starts
dim(train_ds[1]$x)
dim(train_ds[1]$y)

# Dataloaders
batch_size <- 128

train_dl <- train_ds |>
  dataloader(batch_size = batch_size, shuffle = TRUE)
valid_dl <- valid_ds |>
  dataloader(batch_size = batch_size)
test_dl <- test_ds |>
  dataloader(batch_size = length(test_ds))

b <- train_dl |>
  dataloader_make_iter() |>
  dataloader_next()

dim(b$x)
dim(b$y)

model <- nn_module(
  initialize = function(input_size,
                        hidden_size,
                        dropout = 0.2,
                        num_layers = 1,
                        rec_dropout = 0) {

    self$num_layers <- num_layers

    self$rnn <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

    self$dropout <- nn_dropout(dropout)
    self$output <- nn_linear(hidden_size, 1)
  },
  forward = function(x) {
    # take output tensor,restrict to last time step
    self$rnn(x)[[1]][, dim(x)[2], ] |>
      self$dropout() |>
      self$output()
  }
)

input_size <- 1
hidden_size <- 32
num_layers <- 2
rec_dropout <- 0.2

library(luz)

model <- model |>
  setup(optimizer = optim_adam, loss = nn_mse_loss()) |>
  set_hparams(
    input_size = input_size,
    hidden_size = hidden_size,
    num_layers = num_layers,
    rec_dropout = rec_dropout
  )

# Learning Rate finder: novidade do Luz
rates_and_losses <- model |>
  lr_finder(train_dl, start_lr = 1e-3, end_lr = 1)

rates_and_losses |> plot()


fitted <- model |>
  fit(
    train_dl,
    epochs = 5,
    valid_data = valid_dl,
    # exemplos de callbacks: outra novidade do luz
    callbacks = list(
      luz_callback_early_stopping(patience = 3),
      luz_callback_lr_scheduler(
        lr_one_cycle,
        max_lr = 0.1,
        epochs = 10,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"
      )
    ),
    verbose = TRUE
  )

luz::luz_save(fitted, "dados/lstm_demand.pt")

# plotando o fit, mais uma novidade!
plot(fitted)

# predizendo resultados
demand_viz <- demand_hourly |>
  dplyr::filter(lubridate::year(Hour) == 2014, lubridate::month(Hour) == 12)

demand_viz_matrix <- demand_viz |>
  tibble::as_tibble() |>
  dplyr::select(Demand) |>
  as.matrix()

viz_ds <- demand_dataset(demand_viz_matrix, n_timesteps)
viz_dl <- viz_ds |>
  dataloader(batch_size = length(viz_ds))

preds <- predict(fitted, viz_dl)
preds <- preds$to(device = "cpu") |>
  as.matrix()

preds <- c(rep(NA, n_timesteps), preds)

pred_ts <- demand_viz |>
  tibble::add_column(forecast = preds * train_sd + train_mean) |>
  tidyr::pivot_longer(-Hour) |>
  tsibble::update_tsibble(key = name)

pred_ts |>
  feasts::autoplot() +
  ggplot2::scale_colour_manual(values = c("#08c5d1", "#00353f")) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "None")

# Voltando para modelos de textos ------------------------------------------

library(torch)
library(tok)
library(luz)

# adaptado daqui: https://mlverse.github.io/luz/articles/examples/text-classification.html

vocab_size <- 20000 # maximum number of items in the vocabulary
output_length <- 500 # padding and truncation length.
embedding_dim <- 128 # size of the embedding vectors

imdb_dataset <- dataset(
  initialize = function(output_length, vocab_size, root, split = "train", download = TRUE) {
    url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    fpath <- file.path(root, "aclImdb")

    # download if file doesn't exist yet
    if (!dir.exists(fpath) && download) {
      # download into tempdir, then extract and move to the root dir
      withr::with_tempfile("file", {
        download.file(url, file)
        untar(file, exdir = root)
      })
    }

    # now list files for the split
    # set.seed(1)
    self$data <- rbind(
      data.frame(
        fname = list.files(file.path(fpath, split, "pos"), full.names = TRUE),
        y = 1
      ),
      data.frame(
        fname = list.files(file.path(fpath, split, "neg"), full.names = TRUE),
        y = 0
      )
    ) |>
    # APENAS PARA A AULA
    dplyr::slice_sample(n = 2000)

    # train a tokenizer on the train data (if one doesn't exist yet)
    usethis::ui_info("training tokenizer...")
    tokenizer_path <- file.path(root, glue::glue("tokenizer-{vocab_size}.json"))
    if (!file.exists(tokenizer_path)) {
      self$tok <- tok::tokenizer$new(tok::model_bpe$new())
      self$tok$pre_tokenizer <- tok::pre_tokenizer_whitespace$new()

      files <- list.files(file.path(fpath, "train"), recursive = TRUE, full.names = TRUE)
      self$tok$train(files, tok::trainer_bpe$new(vocab_size = vocab_size))
      self$tok$save(tokenizer_path)
    } else {
      self$tok <- tok::tokenizer$from_file(tokenizer_path)
    }

    self$tok$enable_padding(length = output_length)
    self$tok$enable_truncation(max_length = output_length)
  },
  .getitem = function(i) {
    item <- self$data[i, ]

    # takes item i, reads the file content into a char string
    # then makes everything lower case and removes html + punctuaction
    # next uses the tokenizer to encode the text.
    text <- item$fname |>
      readr::read_file() |>
      stringr::str_to_lower() |>
      stringr::str_replace_all("<br />", " ") |>
      stringr::str_remove_all("[:punct:]") |>
      self$tok$encode()

    list(
      x = text$ids + 1L,
      y = item$y
    )
  },
  .length = function() {
    nrow(self$data)
  }
)

train_ds <- imdb_dataset(
  output_length = output_length,
  vocab_size = vocab_size,
  root = "dados/imdb",
  split = "train"
)
test_ds <- imdb_dataset(
  output_length,
  vocab_size,
  "dados/imdb",
  split = "test"
)

train_ds$data |>
  with(fname[1]) |>
  readr::read_file()

model <- nn_module(
  initialize = function(vocab_size, embedding_dim,
                        hidden_lstm = 32, dropout_lstm = 0.2) {

    self$embedding <- nn_sequential(
      nn_embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim),
      nn_dropout(0.1)
    )

    self$lstm <- nn_lstm(
      input_size = embedding_dim,
      hidden_size = hidden_lstm,
      batch_first = TRUE,
      num_layers = 2,
      dropout = dropout_lstm
    )

    self$classifier <- nn_sequential(
      nn_flatten(),
      nn_linear(hidden_lstm, 128),
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(128, 1)
    )
  },
  forward = function(x) {
    emb <- self$embedding(x)
    rnn <- self$lstm(emb)
    out <- rnn[[1]][, dim(emb)[2], ] |>
      self$classifier()
    # we drop the last so we get (B) instead of (B, 1)
    out$squeeze(2)
  }
)

# test the model for a single example batch
# m <- model(vocab_size, embedding_dim)
# x <- torch_randint(1, 20000, size = c(32, 500), dtype = "int")
# m(x)

fitted_model <- model |>
  setup(
    loss = nnf_binary_cross_entropy_with_logits,
    optimizer = optim_adam,
    metrics = luz_metric_binary_accuracy_with_logits()
  ) |>
  set_hparams(
    vocab_size = vocab_size,
    embedding_dim = embedding_dim,
    hidden_lstm = 16,
    dropout_lstm = 0
  ) |>
  fit(train_ds, epochs = 1)

fitted_model |>
  evaluate(test_ds)
