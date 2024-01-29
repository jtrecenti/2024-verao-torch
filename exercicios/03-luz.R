# Exercício 1: Criação de Dataset e DataLoader
# Crie um dataset utilizando o conjunto de dados 'mtcars' e
# defina um DataLoader com um tamanho de batch de 4.

# Bibliotecas necessárias
library(torch)
library(torchvision)

# Conjunto de dados 'cars_scale'
mtcars_scale <- scale(mtcars)

# Criação do Dataset e DataLoader
# Substitua os "##" com o código correto.
ds_mtcars <- ##(mtcars_scale)
dl_mtcars <- ##(ds_mtcars, batch_size = 10)

# Verifique o tamanho do seu dataset e dataloader
print(length(ds_cars))
print(length(dl_cars))


# Exercício 2: Construção de uma MLP
# Construa uma MLP com uma camada oculta de 8 neurônios e uma função de
# ativação ReLU. Ajuste usando o Luz. Utilize mtcars ou outra base de sua
# preferência.

# Exercício 5: Visualização de Predições
# Visualize as predições do seu modelo em um gráfico,
# comparando com os dados reais.

# Exercício 6: Construção de uma CNN
# Crie uma CNN  para a base torchvision::kmnist_dataset()

ds_kmnist <- torchvision::kmnist_dataset(
  ## estudar os parâmetros
)

