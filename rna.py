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

def rna_geral(formato, func, der):
  f = np.vectorize(func)
  df = np.vectorize(der)
  n_camadas = len(formato) - 1
  x_tam = formato[0]
  y_tam = formato[-1]

  x = np.matrix([0.15,0.40])
  w = [] # matrizes de peso
  b = [] # vetores de bias

  # gerar pesos e bias aleatórios
  for i in range(n_camadas):
    w.append(np.random.rand(formato[i+1], formato[i]))
    b.append(np.random.rand(formato[i+1], 1))

  z = [] # vetores de pre-ativação
  a = [] # vetores de ativação
  a.append(x)

  y = np.zeros(y_tam) # valores de saída
  y_esperado = np.matrix([0.01]) # valores esperados

  for j in range(1, 101):
    print(f"******* Epoca {j} *******")
    print(f"====== Entrada ========")
    for i in range(n_camadas):
      ###### Forward Pass ######

      #print(f"a = {a[i]}")
      print(f"w = {w[i].T}")
      #print(f"b = {b[i]}")

      z.append(np.dot(a[i], w[i].T) + b[i].T)
      #print(f"z = {z[i]}")
      a.append(f(z[i]))
      if(i < n_camadas - 1):
        print(f"====== Camada {i+1} ========")
      else:
        print("====== Saída ========")

    y = a[n_camadas]
    print(f"y = {y}")

    dif = y - y_esperado
    #print(dif)
    #print(np.multiply(dif,dif))
    erro = 0.5*(np.multiply(dif,dif))
    print(f"Erro = {erro}")

    #print(a)
    #print(z)
    ###### Backward Pass ######
    nil = 0.5
    delta = [0]*n_camadas # vetores de erro local
    dL_dw = [0]*n_camadas # matrizes de derivadas parciais dos pesos
    dL_db = [0]*n_camadas # vetores de derivadas parciais dos bias

    
    delta[-1] = np.multiply(a[-1] - y_esperado, df(z[-1]))
    dL_dw[-1] = np.dot(a[-2].T,delta[-1])
    #print(f"antigo w2 = {w[-1]}")
    w[-1] = w[-1].T - nil*dL_dw[-1]
    
    #print(f"novo w2 = {w[-1]}")

    dL_db[-1] = delta[-1]
    b[-1] = b[-1] - nil*dL_db[-1]
    for i in range(n_camadas-2, -1, -1):
      
      #print(f"delta{i+1} = {delta[i+1]}")
      #print(f"w = {w[i+1]}")
      #print(f"deriv = {df(z[i])}")

      delta_w = np.matrix(np.dot(w[i+1], delta[i+1]))
      #print(f"delta_w = {delta_w}")
      delta[i] = np.multiply(delta_w, df(z[i]).T)
      
      dL_dw[i] = np.dot(delta[i],a[i])
      #print(f"antigo w{i} = {w[i]}")
      w[i] = w[i] - nil*dL_dw[i]
      #print(f"novo w{i} = {w[i]}")

      dL_db[i] = delta[i]
      #print(f"antigo b{i} = {b[i]}")
      b[i] = b[i] - nil*dL_db[i]
      #print(f"novo b{i} = {b[i]}")

      #print(f"delta{i} = {delta[i]}")
      print("==================") 
    w[-1] = w[-1].T
    z = [] 
    a = []
    a.append(x)
    


formato = [2,3,3,1]

rna_geral(formato, tanh, tanh_deriv)
