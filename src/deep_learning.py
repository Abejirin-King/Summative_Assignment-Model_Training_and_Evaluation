import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_tf_datasets(X_train,y_train,X_val,y_val,batch_size=128):
    train=tf.data.Dataset.from_tensor_slices((X_train.astype("float32"),y_train.astype("float32")))
    train=train.shuffle(min(len(X_train),10000),seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val=tf.data.Dataset.from_tensor_slices((X_val.astype("float32"),y_val.astype("float32")))
    val=val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train,val

def build_mlp(input_dim,units=(128,64,32),dropout=0.30,learning_rate=1e-3):
    inputs=keras.Input(shape=(input_dim,))
    x=inputs
    for u in units:
        x=layers.Dense(u,activation="relu")(x)
        x=layers.BatchNormalization()(x)
        x=layers.Dropout(dropout)(x)
    outputs=layers.Dense(1,activation="sigmoid")(x)
    model=keras.Model(inputs,outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall"),
                           keras.metrics.AUC(name="roc_auc")])
    return model

def train_mlp(model,train_ds,val_ds,epochs=80):
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_roc_auc",mode="max",patience=10,restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=4,min_lr=1e-5)]
    return model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=callbacks,verbose=1)
