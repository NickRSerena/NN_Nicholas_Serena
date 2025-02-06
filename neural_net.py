import numpy as np
import layer


class NeuralNet:

    def __init__(self, loss_func: str = None, display: bool = False, *objects: layer.layer):
        if loss_func is None:
            raise ValueError("Please enter the loss function")
        
        self.layers = objects
        self.loss_func = loss_func
        self.display = display
        self.layer_count = len(self.layers)
        
        

    def params(self, m: int):
        np.random.seed(42)
        self.layers[1].W = np.random.rand(self.layers[1].rows, m)
        self.layers[1].B = np.random.rand(self.layers[1].rows, 1)
        for i in range(2, len(self.layers)):
            self.layers[i].W = np.random.rand(self.layers[i].rows, self.layers[i-1].rows)
            self.layers[i].B = np.random.rand(self.layers[i].rows, 1)

    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true)/y_true.size

    def get_predictions(self, y_pred):
        return np.argmax(y_pred)
        

    def foward(self, X: np.ndarray):
        self.layers[0].Z = X
        self.layers[0].A = X
        for i in range(1, len(self.layers)):
            self.layers[i].dot(self.layers[i-1].Z)
            self.layers[i].act()

        
        return self.layers[-1].A
            

    def backward(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.layers[-1].DZ = self.layers[-1].A - y
        self.layers[-1].DW = 1/n * np.dot(self.layers[-1].DZ, self.layers[-2].A.T)
        self.layers[-1].DB = 1/n * np.sum(self.layers[-1].DZ, axis=1, keepdims=True)

        for i in range(len(self.layers)-2, 0, -1):
            
            self.layers[i].DZ = np.dot(self.layers[i+1].W.T, self.layers[i+1].DZ) * self.layers[i].der_act()
            self.layers[i].DW = 1/n * np.dot(self.layers[i].DZ, self.layers[i-1].A.T)
            self.layers[i].DB = 1/n * np.sum(self.layers[i].DZ, axis=1, keepdims=True)

    def update(self, alpha: float):
        for i in range(1, len(self.layers)):
            self.layers[i].W = self.layers[i].W - alpha * self.layers[i].DW
            self.layers[i].B = self.layers[i].B - alpha * self.layers[i].DB

        
            
       


    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, alpha: float = 0.1, batch_size: int = 10):
        m, n = X.shape
        self.params(m)
        for epoch in range(epochs):
            #randomizes the order of the samples
            I = np.random.permutation(n)

            X_a =  X[:, I] 
            y_a = y[:, I]

            for start in range(0 , n, batch_size):
                end = start + batch_size
                X_b = X_a[:, start:end] # selects the group of samples that are going to be pushed through net
                y_b = y_a[:, start:end] 

                pred =self.foward(X_b)
                self.backward(X_b, y_b)
                self.update(alpha)
                if epoch % 100 == 0 and self.display == True:
                    print("Epoch: ",epoch)
                    print("Pred: ",pred.T," True: ", y.T)
                    print(self.accuracy(pred, y)*100,"%")
                    print()
        

    def test(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.foward(X)
        if self.display == True:
            print(y_pred)
        acc = self.accuracy(y_pred, y)
        if self.display == True:
            print(f'(Accuracy: {acc * 100}%)')




        
        



    def printLayers(self):
        pass


    def get_weights(self, layer):
        return self.layers[layer].W
    
    def get_all_weights(self):
        weights = []
        for l in self.layers:
            weights.append(l.W)

        return weights
    

    def get_middle_layer(self):
        middle = self.layer_count/2
        return self.layers[7]
        


