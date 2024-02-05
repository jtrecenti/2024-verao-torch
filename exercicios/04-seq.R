## OBS: Lista gerada pelo chatgpt. Pode conter erros

# Exercício 1: Criação e Uso de Embeddings
# ----------------------------------------
# Crie um módulo de embedding para um vocabulário de tamanho 10 e dimensão de embedding 4.
# Em seguida, passe um tensor de input com inteiros aleatórios entre 1 e 10 pelo módulo de embedding.
# Dica: Use `nn_embedding` para criar o módulo e `torch_tensor` para criar o tensor de input.
# Exemplo de código incompleto:
vocab_size <- 10
embedding_dim <- 4
# embedding <- nn_embedding(num_embeddings = ?, embedding_dim = ?)
# input_tensor <- torch_tensor(sample(1:?, ?, replace = TRUE), dtype = torch_long())
# print(embedding(input_tensor))

# Exercício 2: Padding de Sequências
# -----------------------------------
# Escreva uma função que aplique padding em um conjunto de sequências para que todas tenham o mesmo comprimento.
# Use 0 como valor de padding.
# Dica: Encontre o comprimento máximo entre as sequências e use `nnf_pad` para aplicar o padding.
# Sequências exemplo: list(c(1,2,3), c(4,5), c(6))
# Exemplo de código incompleto:
sequences <- list(c(1,2,3), c(4,5), c(6))
# pad_sequences <- function(sequences) {
#   max_length <- max(sapply(sequences, length))
#   lapply(sequences, function(seq) { ... })
# }
# print(pad_sequences(sequences))

# Exercício 3: Construção de um Modelo Simples com RNN
# -----------------------------------------------------
# Defina um módulo nn_module que inclua uma camada de embedding seguida por uma RNN simples.
# O módulo deve aceitar sequências de inteiros, aplicar embedding e passar o resultado pela RNN.
# Dica: Inicialize a camada de embedding dentro de `initialize` e defina o forward pass.
# Exemplo de código incompleto:
# model <- nn_module(
#   "SimpleRNNModel",
#   initialize = function(vocab_size, embedding_dim, hidden_size) {
#     self$embedding <- nn_embedding(num_embeddings = ?, embedding_dim = ?)
#     self$rnn <- nn_rnn(input_size = ?, hidden_size = ?, batch_first = TRUE)
#   },
#   forward = function(x) {
#     x <- self$embedding(x)
#     self$rnn(x)
#   }
# )

# Exercício 4: LSTM para Previsão de Séries Temporais
# ---------------------------------------------------
# Utilize uma LSTM para modelar uma série temporal simples. A série pode ser um tensor randômico simulando valores de ações.
# Defina um nn_module com uma camada LSTM seguida por uma camada linear para predição.
# Dica: A camada LSTM deve ser seguida por um reshape ou operação similar para adequar as dimensões ao linear.
# Exemplo de código incompleto:
# time_series_model <- nn_module(
#   "TimeSeriesModel",
#   initialize = function(input_size, hidden_size) {
#     self$lstm <- nn_lstm(input_size = ?, hidden_size = ?, batch_first = TRUE)
#     self$linear <- nn_linear(hidden_size, 1)
#   },
#   forward = function(x) {
#     x <- self$lstm(x)
#     x[[1]] <- ...
#     self$linear(...)
#   }
# )

# Exercício 5: Classificação de Texto com Embeddings e RNN
# --------------------------------------------------------
# Construa um modelo para classificação de texto que utilize embeddings e uma RNN.
# O modelo deve aceitar sequências de inteiros (ids de palavras), aplicar embeddings, e usar a RNN para classificação.
# Dica: Após a RNN, utilize uma camada linear para obter a saída de classificação.
# Exemplo de código incompleto:
# text_classification_model <- nn_module(
#   "TextClassificationModel",
#   initialize = function(vocab_size, embedding_dim, hidden_size) {
#     self$embedding <- nn_embedding(num_embeddings = ?, embedding_dim = ?)
#     self$rnn <- nn_rnn(input_size = ?, hidden_size = ?, batch_first = TRUE)
#     self$linear <- nn_linear(hidden_size, 1) # Assumindo classificação binária
#   },
#   forward = function(x) {
#     x <- self$embedding(x)
#     x <- self$rnn(x)
#     x[[1]] <- ...
#     torch_sigmoid(self$linear(x[[1]]))
#   }
# )

