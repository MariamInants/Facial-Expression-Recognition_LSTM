from .BaseNN import *
class DNN(BaseNN):
      
   def network(self, X):

        W = {
            'hidden': tf.Variable(tf.random_normal([self.width_of_image, 30])),
            'output': tf.Variable(tf.random_normal([30, self.num_classes]))
        }
        b = {
            'hidden': tf.Variable(tf.random_normal([30], mean=1.0)),
            'output': tf.Variable(tf.Variable(tf.random_normal([self.num_classes])))
        }

        # Transpose and then reshape to 2D of size (BATCH_SIZE * SEGMENT_TIME_SIZE, N_FEATURES)
        X = tf.transpose(X, [1, 0, 2])
        X = tf.reshape(X, [-1, self.width_of_image])

        hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + b['hidden'])
        hidden = tf.split(hidden, self.height_of_image, 0)

        # Stack two LSTM cells on top of each other
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(30, forget_bias=1.0)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(30, forget_bias=1.0)
        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2])

        outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

        # Get output for the last time step from a "many to one" architecture
        last_output = outputs[-1]

        return tf.matmul(last_output, W['output'] + b['output'])
       
   def metrics(self, Y, Y_pred):
       y_pred_softmax = tf.nn.softmax(Y_pred, name="y_pred_softmax")
       correct_pred = tf.equal(tf.argmax(y_pred_softmax, 1), tf.argmax(Y, 1))
       # tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
       loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
       # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
       # loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(Y,Y_pred))
       
       accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

       # tf.reduce_mean(tf.cast(correct_pred, tf.float32))
       return loss, accuracy , y_pred_softmax

