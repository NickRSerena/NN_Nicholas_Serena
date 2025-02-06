import numpy as np
class layer:
    def __init__(self, nodes: int = None, activation: str = 'input'):
        if nodes is None:
            raise ValueError("Please enter the number of nodes in the layer")
        
        activation_func = {'relu': (self.relu, self.derivative_relu), 
                           'sigmoid': (self.sigmoid, self.derivative_sigmoid), 
                           'softmax': (self.softmax, self.softmax)} # dictionary for all the activation functions and the derivatives
        
        self.rows = nodes
        

        self.W = None
        self.DW = None

        self.B = None
        self.DB = None

        self.Z = np.zeros((nodes, 1))
        self.DZ = None

        self.A = None

       
        if activation == 'relu':
            self.act = activation_func['relu'][0]
            self.der_act = activation_func['relu'][1]
        elif activation == 'sigmoid':
            self.act = activation_func['sigmoid'][0]
            self.der_act = activation_func['sigmoid'][1]
        elif activation == 'softmax':
            self.act = activation_func['softmax'][0]
            self.der_act = activation_func['softmax'][1]


        





    def dot(self, x: np.ndarray):
        self.Z = self.W @ x + self.B

    def softmax(self):
        e_z = np.exp(self.Z)
        sum_ez = np.sum(e_z, axis = 1, keepdims= True)
        self.A = e_z/sum_ez
        

    def relu(self):
        self.A = np.maximum(0, self.Z)

    def derivative_relu(self):
        return np.where(self.Z > 0, 1.0, 0.0)
    
    def sigmoid(self):
        self.A = 1 / (1 + np.exp(-self.Z))

    def derivative_sigmoid(self):
        return self.Z * (1 - self.Z)
