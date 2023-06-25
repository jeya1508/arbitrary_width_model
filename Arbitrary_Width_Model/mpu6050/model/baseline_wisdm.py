def wisdm_cnn(x_train,y_train,x_test,y_test):
            channels=[128, 256, 512, 23040]
            
            model= keras.Sequential()
            model.add(Input(shape=(200,3,1)))
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[0], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[1], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.Conv2D(channels[2], (3,3),strides=(2,2), activation='relu',padding='valid'))
            model.add(layers.BatchNormalization())
            model.add(layers.ZeroPadding2D(padding=(1,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(1,1), padding='valid'))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(6,activation='softmax'))
            model.summary()
            #return model
            batch_size=128
            #epochs=25
            
            
            #x_train = x_train.reshape(-1, 1, 128, 9)
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics=["accuracy"])
            model.fit(x_train,y_train,batch_size=batch_size,epochs=args.epoch,validation_split=0.1)
            scores = model.evaluate(x_test, y_test, verbose=0) 
            print("Accuracy: %.2f%%" % (scores[1]*100))