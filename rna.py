import numpy as np

def sigmoide(z):
  return 1/(1 + np.exp(-z))

def sig_deriv(z):
  return sigmoide(z)*(1-sigmoide(z))

def relu(z):
  return max(0, z)

def relu_deriv(z):
  if z > 0:
    return 1
  else:
    return 0

def tanh(z):
  return np.tanh(z)

def tanh_deriv(z):
  return 1 - np.tanh(z)**2

def rna_geral(formato, func):
  f = np.vectorize(func)

  n_camadas = len(formato) - 1
  x_tam = formato[0]
  y_tam = formato[-1]

  x = np.matrix(np.random.rand(formato[0]))
  w = [] # matrizes de peso
  b = [] # vetores de bias

  # gerar pesos e bias aleatórios
  for i in range(n_camadas):
    w.append(np.random.rand(formato[i+1], formato[i]))
    b.append(np.random.rand(formato[i+1], 1))

  #z = np.zeros(n_camadas) # vetores de pre-ativação
  z = []
  a = [] # vetores de ativação
  a.append(x)

  y = np.zeros(y_tam) # valores de saída
  y_esperado = np.matrix([0.01,0.02]) # valores esperados

  print(f"====== Entrada ========")
  for i in range(n_camadas):
    ###### Forward Pass ######

    print(f"a = {a[i]}")
    print(f"w = {w[i]}")
    print(f"b = {b[i]}")

    z.append(np.dot(a[i], w[i].T) + b[i].T)
    print(f"z = {z[i]}")
    a.append(f(z[i]))
    if(i < n_camadas - 1):
      print(f"====== Camada {i+1} ========")
    else:
      print("====== Saída ========")
  y = a[n_camadas]
  print(f"y = {y}")
#print(z)
#erro = 0.5*(y - y_esperado)**2

'''
###### Backward Pass ######
nil = 0.5

## Camada 2
delta2 = (a2 - y_esperado)*(1 - a2)*a2
d2w = delta2*a1
d2b = delta2

# atualizacao dos parametros
w2 = w2 - nil*d2w
b2 = b2 - nil*d2b

## Camada 1
delta1 = delta2*w2*a1*(1-a1)
d1w = np.outer(delta1, x)
d1b = delta1

# atualizacao dos parametros
w1 = w1 - nil*d1w
b1 = b1 - nil*d1b
'''

formato = [2,3,3,2]

rna_geral(formato, tanh)
