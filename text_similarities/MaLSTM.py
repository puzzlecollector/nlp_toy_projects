## Simple Demo Code using Tensorflow 2.0 
## MaLSTM model for finding text similarity 

class Model(tf.keras.Model):
    def __init__(self, **kargs):
        super(Model, self).__init__(name = model_name)
        self.embedding = layers.Embedding(input_dim = kargs['vocab_size'],
                                          output_dim = kargs['embedding_dimension'])
        self.lstm = layers.LSTM(units = kargs['lstm_dimension'])

    def call(self, x):
        x1, x2 = x # x1,x2 are the two texts
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = self.lstm(x1)
        x2 = self.lstm(x2)
        x = tf.exp(-tf.reduce_sum(tf.abs(x1-x2), axis = 1))
        return x
