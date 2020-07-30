import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten
from tensorflow.keras import Model
#%%
class Residual(Layer):
    
    def __init__(self, nFilters=128, filterSize=3):
        super(Residual, self).__init__()
        
        self.conv1 = Conv2D(filters=nFilters, kernel_size=filterSize, padding='same')
        self.batchNorm1 = BatchNormalization()
        
        self.conv2 = Conv2D(filters=nFilters, kernel_size=filterSize, padding='same')
        self.batchNorm2 = BatchNormalization()
        
    def call(self, input):
        x = self.conv1(input)
        x = self.batchNorm1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.batchNorm2(x)
        
        #skip layer: concatenate x and input along the nFilters axis (may need to change axis)
        x = tf.concat([x, input], axis=3)
        y = tf.nn.relu(x)
        return y

#%%    
class FirstConv(Layer):
    
    def __init__(self, nFilters=128, filterSize=3):
        super(FirstConv, self).__init__()
        
        self.conv = Conv2D(filters=nFilters, kernel_size=filterSize, padding='same')
        self.batchNorm = BatchNormalization()
        
    def call(self, input):
        x = self.conv(input)
        x = self.batchNorm(x)
        y = tf.nn.relu(x)
        return y
    
#%%
class ValueHead(Layer):
    
    def __init__(self, denseUnits=256, dropoutRate=0.5):
        super(ValueHead, self).__init__()
        
        self.dropoutRate = dropoutRate
        
        #condense the depth into a single image
        self.condensingConv = Conv2D(filters=1, kernel_size=1)
        self.batchNorm = BatchNormalization()
        
        self.flatten = Flatten()        
        self.dense1 = Dense(units=denseUnits)
        
        self.dense2 = Dense(units=1)
        
    def call(self, input):
        
        x = self.condensingConv(input)
        x = self.batchNorm(x)
        x = tf.nn.relu(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.nn.relu(x)
        
        x = tf.nn.dropout(x, rate=self.dropoutRate)
        
        x = self.dense2(x)
        y = tf.nn.tanh(x)
        
        return y
    
#%%
class ValueNetwork(Model):
    
    def __init__(self, nResidualLayers=3,nFilters=128, 
                 filterSize=3, denseUnits=256, dropoutRate=0.5):
        super(ValueNetwork, self).__init__()
        
        self.firstConv = FirstConv(nFilters, filterSize)
        self.residuals = [Residual(nFilters, filterSize) for _ in range(nResidualLayers)]
        self.valueHead = ValueHead(denseUnits, dropoutRate)
        
    def call(self, input):
        x = self.firstConv(input)
        for i in range(len(self.residuals)):
            x = self.residuals[i](x)
            
        y = self.valueHead(x)
        return y