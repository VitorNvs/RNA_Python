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

class NeuralMLP():
  def __init__(self, shape, func, dfunc):
    self.w = [] # matrizes de peso
    self.b = [] # vetores de bias
    self.z = [] # vetores de pre-ativação
    self.a = [] # vetores de ativação

    self.f = np.vectorize(func)
    self.df = np.vectorize(dfunc)
    self.n_layers = len(shape) - 1

    # gerar pesos e bias aleatórios
    for i in range(self.n_layers):
      self.w.append(np.random.rand(shape[i+1], shape[i]))
      self.b.append(np.random.rand(shape[i+1], 1))
    
    self.x_size = shape[0]
    self.y_size = shape[-1]


  def mlpTraining(self, x, y, epochs):
    x = np.matrix(x)
    y = np.matrix(y)
    self.a = []
    self.a.append(x)

    for j in range(epochs):
      print(f"******* Epoca {j+1} *******")
      print(f"====== Entrada ========")

      for i in range(self.n_layers):
        ###### Forward Pass ######

        self.z.append(np.dot(self.a[i], self.w[i].T) + self.b[i].T)
        self.a.append(self.f(self.z[i]))
        if(i < self.n_layers - 1):
          print(f"====== Camada {i+1} ========")
        else:
          print("====== Saída ========")

      y_ = self.a[self.n_layers]
      print(f"y = {y_}")

      dif = y_ - y
      erro = 0.5*(np.multiply(dif,dif))
      print(f"Erro = {erro}")

      ###### Backward Pass ######
      nil = 0.5
      delta = [0]*self.n_layers # vetores de erro local
      dL_dw = [0]*self.n_layers # matrizes de derivadas parciais dos pesos
      dL_db = [0]*self.n_layers # vetores de derivadas parciais dos bias

      # Erro local da última camada
      delta[-1] = np.multiply(self.a[-1] - y, self.df(self.z[-1]))

      # Derivação e atualização dos pesos da última camada
      dL_dw[-1] = np.dot(self.a[-2].T,delta[-1])
      self.w[-1] = self.w[-1].T - nil*dL_dw[-1]
      
      # Derivação e atualização dos bias da última camada
      dL_db[-1] = delta[-1]
      self.b[-1] = self.b[-1] - nil*dL_db[-1]

      for i in range(self.n_layers-2, -1, -1):
        # Erro local
        delta_w = np.matrix(np.dot(self.w[i+1], delta[i+1]))
        delta[i] = np.multiply(delta_w, self.df(self.z[i]).T)
        
        # Derivação e atualização dos pesos
        dL_dw[i] = np.dot(delta[i],self.a[i])
        self.w[i] = self.w[i] - nil*dL_dw[i]

        # Derivação e atualização dos bias
        dL_db[i] = delta[i]
        self.b[i] = self.b[i] - nil*dL_db[i]
        
        print("==================") 
      self.w[-1] = self.w[-1].T
      self.z = [] 
      self.a = []
      self.a.append(x)
      print(f"w = {self.w}")

  def generateOutput(self, x):
    print("============= Gerando Saída =============")
    ###### Forward Pass ######
    for i in range(self.n_layers):
      self.z.append(np.dot(self.a[i], self.w[i].T) + self.b[i].T)
      self.a.append(self.f(self.z[i]))

    y_ = self.a[self.n_layers]
    print(f"y = {y_}")


formato = [2,3,3,1]
entrada1 = [0.15,0.40]
saida1 = [0.02]

entrada2 = [0.50,0.30]
saida2 = [0.05]

rna = NeuralMLP(formato, tanh, tanh_deriv)
rna.mlpTraining(entrada1,saida1,100)
#rna.mlpTraining(entrada2,saida2,100)

rna.generateOutput(entrada1)

