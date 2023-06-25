def to_one_hot(y, no_of_dims=None):
    
    y = tf.cast(y, dtype=tf.int64)
    y_value = tf.reshape(y, [-1, 1])
    no_of_dims = no_of_dims if no_of_dims is not None else tf.cast(tf.reduce_max(y) + 1, tf.int32)
    zeros_tensor = tf.zeros([y_value.shape[0], no_of_dims])
    updates = tf.ones([y_value])
    indices = tf.stack([tf.range(batch_size), y_value], axis=1)
    ones_tensor = tf.tensor_scatter_nd_update(zeros_tensor, indices, updates)
    y_one_hot = tf.reshape(y_one_hot, [*y.shape, -1])
    return y_one_hot

def adjust_learn_rate(optim, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50))  
    for var in optimizer.variables():
        if 'learning_rate' in var.name:
            var.assign(lr)
        
       
def train(model, train_batch, loss_fn, optim, learn_rate, epoch):
    adjust_learn_rate(optim, epoch, learn_rate)
    accuracy = tf.keras.metrics.Accuracy()
    model.trainable=True
    train_loss_metric = tf.keras.metrics.Mean()
    loss_epoch = 0
    acc_epoch = 0
    loss_list = []

    for step, (x, y) in enumerate(train_batch):
        train_loss_metric.reset_states()
        #inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
        batch_x = tf.cast(batch_x, tf.float32)
        with tf.device('/GPU:0'):
            inputs = tf.Variable(batch_x)
        batch_y = tf.cast(batch_y, tf.int64)
        with tf.device('/GPU:0'):
            labels = tf.Variable(batch_y)

        labels_conv_onehot = to_one_hot(labels)
        with tf.GradientTape() as tape:
            
            # Forward pass
            y_pred = model(inputs)
            preds = tf.argmax(outputs, axis=1)
            accuracy.update_state(labels, preds)
            epoch_acc = accuracy.result().numpy()
            # Compute loss
            loss = loss_fn(y_pred,labels)

        # Compute gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # Update model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_loss = train_loss / len(train_batch)
        # Clear gradients
        tape.reset()
        

        print("Training epoch {}, Loss {}, Accuracy {}, Learning Rate {}".format(step, loss.numpy(),accuracy,optim.learning_rate.numpy()))

    return train_loss, epoch_acc



def test(model, test_batch, loss_fn):  
    test_loss_metric = tf.keras.metrics.Mean()
    epoch_loss= 0
    epoch_acc = 0
    accuracy = tf.keras.metrics.Accuracy()
    conf_matrix = np.array(0)
    num_batches=0
    model.eval()
    
    for step, (x,y) in enumerate(test_batch):
        batch_x = tf.cast(batch_x, tf.float32)
        with tf.device('/GPU:0'):
            inputs = tf.Variable(batch_x)
        batch_y = tf.cast(batch_y, tf.int64)
        with tf.device('/GPU:0'):
            labels = tf.Variable(batch_y)
        with tf.GradientTape() as tape:
        #inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            y_pred = model(inputs)
            preds = tf.argmax(outputs, axis=1)
            accuracy.update_state(labels, preds)
            loss = loss_fn(y_pred,labels)
            #batch_loss = compute_loss(batch_x, batch_y)
            epoch_loss += loss.numpy()
            num_batches += 1
        grads = tape.gradient(loss, model.trainable_variables)
        # Update model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        test_loss_metric.update_state(loss)
        test_loss = test_loss / len(test_batch)
        # Clear gradients
        tape.reset()

    
    return epoch_loss, epoch_acc, conf_matrix