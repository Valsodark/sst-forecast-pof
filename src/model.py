from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras import regularizers

def create_model(history, patch_h, patch_w):
    l2 = 1e-5
    model = Sequential([
        ConvLSTM2D(
            8, (3, 3), padding='same', return_sequences=True,
            activation='tanh',
            input_shape=(history, patch_h, patch_w, 1),
            kernel_regularizer=regularizers.l2(l2),
            recurrent_regularizer=regularizers.l2(l2),
            dropout=0.12, recurrent_dropout=0.04,
        ),
        BatchNormalization(),
        ConvLSTM2D(
            4, (3, 3), padding='same', return_sequences=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(l2),
            recurrent_regularizer=regularizers.l2(l2),
            dropout=0.12, recurrent_dropout=0.04,
        ),
        BatchNormalization(),
        Conv2D(1, (3, 3), padding='same', activation='linear', dtype='float32')
    ])
    return model

def compile_model(model, initial_lr=5e-4):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=2000,
        decay_rate=0.96
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model