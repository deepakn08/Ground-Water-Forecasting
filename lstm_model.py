import tensorflow as tf, numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger

def set_seed(seed_no):
    np.random.seed(seed_no)
    tf.random.set_seed(seed_no)

def predict_distribution(X, model, n):
    preds = [model(X) for _ in range(n)]
    # Convert to a 3D array, the shape is currently (n, 269, 3)
    preds = np.array(preds)
    # Now reshape to (269, 3, n)
    preds = preds.transpose(1, 2, 0)
    return preds

class MCDropout(tf.keras.layers.Dropout):
    #define Monte Carlo Dropout Layer, where training state is always true (even during prediction)
    def call(self, inputs):
        return super().call(inputs, training=True)

class StackLSTMModel(tf.keras.Model):
    def __init__(   self,
                    seq_length,
                    num_features,
                    num_outputs             =   1,
                    no_of_lstm_layers       =   2,
                    lstm_hidden_units       =   128,
                    dense_units_list        =   [64, 32],
                    bidirectional           =   True,
                    use_state_layernorms    =   2,
                    state_management        =   True,
                    dense_activation_layer  =   'relu'
                ):
        super(StackLSTMModel, self).__init__()

        self.bidirectional = bidirectional
        self.no_of_lstm_layers = no_of_lstm_layers
        self.lstm_hidden_units = lstm_hidden_units
        self.dense_units_list = dense_units_list

        assert use_state_layernorms in [0, 1, 2], 'use_state_layernorm should be either 0, 1, or 2'
        assert state_management in [True, False], 'state_management should be either True or False'
        self.use_state_layernorms = use_state_layernorms
        self.state_management = state_management
        self.state_layernorms_for_each_layer = 0
        if self.state_management and self.use_state_layernorms:
                self.state_layernorms_for_each_layer = (    (4 if self.bidirectional else 2)
                                                                if self.use_state_layernorms == 2 else
                                                            1   )

        # Create LSTM layers
        self.lstm_layers = []
        self.lstm_layernorms = []
        self.lstm_state_layernorms = []

        # Create the first LSTM layer with input shape
        self._add_lstm_layer(input_shape = (seq_length, num_features))
        for _ in range(self.no_of_lstm_layers - 1):     # Create remaining LSTM layers
            self._add_lstm_layer()

        # Create Dense layers
        self.dense_layers = [tf.keras.layers.Flatten(), MCDropout(0.5)]
        self._build_dense_layer_stack(num_outputs, dense_activation_layer)  # Build the dense layers
        self.dense_layers = tf.keras.Sequential(self.dense_layers)  # After defining the dense layers turn it into sequential

    def _add_lstm_layer(self, input_shape=None):

        if input_shape:
            lstm_layer = tf.keras.layers.LSTM(self.lstm_hidden_units,
                                                return_sequences=True,
                                                return_state=True,
                                                input_shape=input_shape)
        else:
            lstm_layer = tf.keras.layers.LSTM(self.lstm_hidden_units,
                                                return_sequences=True,
                                                return_state=True)
        if self.bidirectional:
            lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)
        self.lstm_layers.append(lstm_layer)

        # LayerNorm after each LSTM
        self.lstm_layernorms.append(tf.keras.layers.LayerNormalization())
        state_layernorms = None
        if self.state_management and self.use_state_layernorms:
            state_layernorms = (    [   tf.keras.layers.LayerNormalization()
                                            for _ in range(self.state_layernorms_for_each_layer)    ]

                                        if self.use_state_layernorms == 2 else

                                    tf.keras.layers.LayerNormalization()    )

        self.lstm_state_layernorms.append(state_layernorms)

    def _add_dense_layer(self, units, normalize=False, activation=None):
        self.dense_layers.append(tf.keras.layers.Dense(units))
        if normalize:
            self.dense_layers.append(tf.keras.layers.LayerNormalization())
        if activation:
            self.dense_layers.append(tf.keras.layers.Activation(activation))

    def _build_dense_layer_stack(self, num_outputs, activation_layer):

        # Add the first dense layer
        flatten_linear_output_features = self.dense_units_list[0] if self.dense_units_list else num_outputs
        self._add_dense_layer(flatten_linear_output_features)
        if self.dense_units_list:
            self.dense_layers.append(tf.keras.layers.LayerNormalization())
            self.dense_layers.append(tf.keras.layers.Activation(activation_layer))

        # Add remaining dense layers
        for dense_size in self.dense_units_list[1:]:
            self._add_dense_layer(dense_size, normalize=True, activation=activation_layer)

        # Add final output layer if needed
        if flatten_linear_output_features != num_outputs:
            self._add_dense_layer(num_outputs)

    def call(self, x):

        # list for explicit state management of LSTM
        lstm_units_states = []  # To hold the states of each LSTM layer

        for layer, (layernorm, state_layernorms) in zip(self.lstm_layers, zip(self.lstm_layernorms, self.lstm_state_layernorms)):
            outputs = None
            if self.state_management and lstm_units_states:
                outputs = layer(x, initial_state=lstm_units_states[-1])     # Use the last state from the previous layer
            else:
                outputs = layer(x)  # First layer or no explicit state management

            x = outputs[0]
            curr_states = list(outputs[1:])     # Possibilities [x, h_f, c_f, h_b, c_b] OR [x, h, c]

            x = layernorm(x)
            if self.state_management and self.use_state_layernorms:
                for i in range(len(curr_states)):
                    curr_states[i] =  ( state_layernorms[i](curr_states[i])
                                            if isinstance(state_layernorms, list) else
                                        state_layernorms(curr_states[i])    )
                lstm_units_states.append(curr_states)  # Store the states for the next layer

        x = self.dense_layers(x)
        return x


# =============================================================================
#### DEFINE YOUR MODEL HERE
# =============================================================================
## IMPORTANT NOTE_
## TO STICK TO ORIGINAL OBJECTIVE ITS PREFERABLE TO USE "MCDropout" LAYER THAN THE NORMAL DROPOUT LAYER

'''
def gwmodel(ini, GLOBAL_SETTINGS,
            X_train, Y_train,
            X_stop, Y_stop,
            verbose = 0):

    #####
    # Try to clean any previous model, which is present in GPU
    tf.keras.backend.clear_session()
    # Set seed for reproducibility
    set_seed(ini+87747)
    #####

    # define model

    input = tf.keras.Input(shape=(GLOBAL_SETTINGS["seq_length"], GLOBAL_SETTINGS["num_features"]))
    x = tf.keras.layers.LSTM(units=GLOBAL_SETTINGS["lstm1_units"], return_sequences = True)(input)
    x = tf.keras.layers.LayerNormalization(axis = 2)(x)
    x = tf.keras.layers.LSTM(units=GLOBAL_SETTINGS["lstm2_units"], return_sequences = True)(x)
    x = tf.keras.layers.LayerNormalization(axis = 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = MCDropout(0.5)(x)
    x = tf.keras.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(x)
    output = tf.keras.layers.Dense(GLOBAL_SETTINGS["prediction_length"], activation='linear')(x)

    # tie together
    model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"],
                                        clipnorm=GLOBAL_SETTINGS["clip_norm"])

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])


    # early stopping
    es = tf.keras.callbacks.EarlyStopping(  monitor     =   GLOBAL_SETTINGS['early_stopping_monitor'],
                                            mode        =   GLOBAL_SETTINGS['early_stopping_mode' ],
                                            patience    =   GLOBAL_SETTINGS['early_stopping_patience'],
                                            restore_best_weights = True,
                                            verbose     =   verbose
                                            #,start_from_epoch = 30
                                        )

    #######
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(   monitor     =   GLOBAL_SETTINGS['reducelr_monitor'],
                                                        mode        =   GLOBAL_SETTINGS['reducelr_mode'],
                                                        factor      =   GLOBAL_SETTINGS['reducelr_factor' ],
                                                        patience    =   GLOBAL_SETTINGS['reducelr_patience'],
                                                        min_lr      =   GLOBAL_SETTINGS['reducelr_minlr'],
                                                        verbose     =   verbose
                                                    )

    # fit network
    history = model.fit(    X_train, Y_train,
                            validation_data=(X_stop, Y_stop),
                            epochs=GLOBAL_SETTINGS["epochs"],
                            batch_size=GLOBAL_SETTINGS["batch_size"],
                            callbacks=[es, reduce_lr],
                            verbose=verbose
                        )

    return model, history
#'''


#'''
def gwmodel(    ini, global_settings,
                X_train, Y_train,
                X_stop, Y_stop,
                verbose = 0
            ):

    #####
    # Try to clean any previous model, which is present in GPU
    tf.keras.backend.clear_session()
    # Set seed for reproducibility
    set_seed(ini)

    if (not global_settings['disable_wandb']) and (global_settings['use_wandb'] == True):
            wandb.finish()
            wandb.init(
                        project=global_settings['project_name'],
                        config=global_settings,
                        group=global_settings['runs_group'],
                        name=global_settings['run_name'],
                        dir= (global_settings['pathResults'] +'/wandb__'),
                    )
    #####

    # define model
    model = StackLSTMModel( seq_length              =   global_settings["seq_length"],
                            num_features            =   global_settings["num_features"],
                            num_outputs             =   global_settings["prediction_length"],
                            no_of_lstm_layers       =   global_settings["no_of_lstm_layers"],
                            lstm_hidden_units       =   global_settings["lstm_hidden_units"],
                            dense_units_list        =   global_settings["dense_units_list"],
                            bidirectional           =   global_settings["bidirectional"],
                            use_state_layernorms    =   global_settings["use_state_layernorms"],
                            state_management        =   global_settings["state_management"],
                            dense_activation_layer  =   global_settings["dense_activation_layer"]
                        )

    optimizer = tf.keras.optimizers.Adam(   learning_rate   =   global_settings["learning_rate"],
                                            clipnorm        =   global_settings["clip_norm"]
                                        )

    model.compile(  loss        =   tf.keras.losses.MSE,
                    optimizer   =   optimizer,
                    metrics     =   [   tf.keras.metrics.MeanSquaredError    ( name = 'MSE' ),
                                        tf.keras.metrics.RootMeanSquaredError( name = 'RMSE'),
                                        tf.keras.metrics.R2Score             ( name = 'NSE' ),
                                        tf.keras.metrics.MeanAbsoluteError   ( name = 'MAE' ),
                                    ]
                )

    # early stopping
    earlystopping = tf.keras.callbacks.EarlyStopping(   monitor     =   global_settings['early_stopping_monitor'],
                                                        mode        =   global_settings['early_stopping_mode' ],
                                                        patience    =   global_settings['early_stopping_patience'],
                                                        restore_best_weights = True,
                                                        verbose     =   verbose
                                                        #,start_from_epoch = 30
                                                    )

    #######
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(   monitor     =   global_settings['reducelr_monitor'],
                                                        mode        =   global_settings['reducelr_mode'],
                                                        factor      =   global_settings['reducelr_factor' ],
                                                        patience    =   global_settings['reducelr_patience'],
                                                        min_lr      =   global_settings['reducelr_minlr'],
                                                        verbose     =   verbose
                                                    )

    callback_list = [earlystopping, reduce_lr]

    wandb_callback = None
    if (not global_settings['disable_wandb']) and (global_settings['use_wandb'] == True):
            wandb_callback = WandbMetricsLogger()
            callback_list.append(wandb_callback)

    # fit network
    history = model.fit(    X_train, Y_train,
                            validation_data     =   (X_stop, Y_stop),
                            epochs              =   global_settings["epochs"],
                            batch_size          =   global_settings["batch_size"],
                            callbacks           =   callback_list,
                            verbose             =   verbose
                        )

    return model, history
#'''

