# Exercício Teórico 1: Entendimento da Regressão Linear
# Explique, com suas próprias palavras, o que a equação
# y = β0 + β1 * x + ε representa no contexto da regressão linear.


# Exercício Teórico 2: Função de Perda e Verossimilhança
# Pergunta: Descreva a relação entre a função de perda do erro quadrático médio (MSE)
# e a verossimilhança na regressão linear.
# Por que minimizar o MSE é equivalente a maximizar a verossimilhança?


# Exercício Prático 3: Cálculo de Gradientes
library(torch)

t1 <- torch_tensor(10, requires_grad = TRUE)
# Realize algumas operações com t1, diferentes das vistas em aula. Exemplo:

t2 <- t1 + 2 # mude isso
t3 <- t2$square() # e isso

# Agora calcule o gradiente
t3$backward()
print(t1$grad)

# Compare o resultado do gradiente calculado manualmente.

# Exercício Teórico 4: Derivação do Gradiente
# Obtenha o gradiente da função de perda MSE em relação a β0 e β1.
# Como isso se relaciona com a atualização dos parâmetros na descida de gradiente?

# Exercício Prático 5: Implementação da Regressão Linear com Autograd
# Use o dataset 'mtcars' para implementar uma regressão linear múltipla
# usando descida de gradiente com autograd.

mtcars_scale <- scale(mtcars)

# Implemente a regressão linear e compare com a solução analítica.

# Exercício Prático 6: MLP com Descida de Gradiente
# Construa e treine uma MLP simples para o dataset 'mtcars'.
# Implemente a MLP e o processo de treinamento aqui.

# Exercício Teórico 7: Compreensão de Otimizadores
# Pergunta: Explique a diferença entre a descida de gradiente simples e
# métodos de otimização como L-BFGS ou Adam.
# Em que situações um pode ser preferível ao outro?
