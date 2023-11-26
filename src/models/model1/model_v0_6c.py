"""
Last update: 2023-10-29
V0.6: each feature has its own embedding layer
"""
import tensorflow as tf
from tensorflow import keras
class customNN(keras.Sequential):
    def __init__(self, 
                 regularizer: str|None, regularizer_weight: int|None,
                 initializer: str|None,
                 dropout_rate: float|None,
                 name,
                 input_size, 
                 output_size,
                 hidden_layer_widths: int|list,
                 hidden_layer_depth: int| None,
                 hidden_layer_activation = 'relu',
                 output_layer_activation = 'linear',
                 masking: bool = False,
                 ):
        super().__init__(name = name)
        """
        when `widths` is an `int`, all the Dense layers are identical in size (width)
        `widths` only indicates the widths in the middle layer, NOT the last layer. 
        The last layer's widths is indicated by `output_size`
        """
        if isinstance (hidden_layer_widths, int) and isinstance (hidden_layer_depth, int):
            # First dense layer or masking layer defined by input shape, 
            if masking: 
                self.add(keras.layers.Masking(mask_value=0, input_shape = (input_size,)))
                self.add(keras.layers.Dense(hidden_layer_widths, activation= hidden_layer_activation))
            else: 
                self.add(keras.layers.Dense(hidden_layer_widths, input_shape=(input_size,), activation= hidden_layer_activation))
            if dropout_rate != None: 
                self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            # Custom layer of hidden Dense layer
            for _ in range (hidden_layer_depth-1):
                self.add (keras.layers.Dense(hidden_layer_widths,activation= hidden_layer_activation))
                if dropout_rate != None: 
                    self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            # Output layer
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        
        elif isinstance (hidden_layer_widths, list) and hidden_layer_depth == None:
            if masking: 
                self.add(keras.layers.Masking(mask_value=0, input_shape = (input_size,)))
                self.add(keras.layers.Dense(hidden_layer_widths[0], activation=hidden_layer_activation))
            else:
                self.add(keras.layers.Dense(hidden_layer_widths[0], input_shape=(input_size,), activation=hidden_layer_activation))
            if dropout_rate is not None: 
                self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            for width in hidden_layer_widths[1:]:
                self.add (keras.layers.Dense(width,activation= hidden_layer_activation))
                if dropout_rate is not None: 
                    self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        else: raise Exception ("CustomNN: widths and depths are set incorrectly.\n Correct cases: (widths is int and depth is int) OR (widths is list and depth is None)")
        
        # APPLYING REGULARIZER: exlude the last Dense layer
        if regularizer == 'l2':
            for layer in self.layers[:-1]:
                layer.kernel_regularizer = tf.keras.regularizers.l2(l2=regularizer_weight)
                
        # INITIALIZER: 
        if initializer == 'he':
            initializer = keras.initializers.he_normal(seed = 13)
            for layer in self.layers: 
                layer.kernel_initializer = initializer
#### 
class customNN_2(keras.Sequential):
    def __init__(self, 
                 regularizer: str|None, regularizer_weight: int|None,
                 initializer: str|None,
                 dropout_rate: float|None,
                 name,
                 output_size,
                 hidden_layer_widths: int|list,
                 hidden_layer_depth: int| None,
                 input_size = None, 
                 hidden_layer_activation = 'relu',
                 output_layer_activation = 'linear',
                 masking: bool = False,
                 ):
        super().__init__(name = name)
        """
        when `widths` is an `int`, all the Dense layers are identical in size (width)
        `widths` only indicates the widths in the middle layer, NOT the last layer. 
        The last layer's widths is indicated by `output_size`
        """
        if isinstance (hidden_layer_widths, int) and isinstance (hidden_layer_depth, int):
            # First dense layer or masking layer defined by input shape, 
            if masking: 
                if input_size is None: 
                    self.add(keras.layers.Masking(mask_value=0))
                    self.add(keras.layers.Dense(hidden_layer_widths, activation= hidden_layer_activation))
                else: 
                    self.add(keras.layers.Masking(mask_value=0, input_shape = (input_size,)))
                    self.add(keras.layers.Dense(hidden_layer_widths, activation= hidden_layer_activation))
            else: 
                if input_size is None:
                    self.add(keras.layers.Dense(hidden_layer_widths, activation= hidden_layer_activation))
                else: 
                    self.add(keras.layers.Dense(hidden_layer_widths, input_shape=(input_size,), activation= hidden_layer_activation))
            if dropout_rate != None: 
                self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            # Custom layer of hidden Dense layer
            for _ in range (hidden_layer_depth-1):
                self.add (keras.layers.Dense(hidden_layer_widths,activation= hidden_layer_activation))
                if dropout_rate != None: 
                    self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            # Output layer
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        
        elif isinstance (hidden_layer_widths, list) and hidden_layer_depth == None:
            if masking: 
                if input_size is None:
                    self.add(keras.layers.Masking(mask_value=0))
                    self.add(keras.layers.Dense(hidden_layer_widths[0], activation=hidden_layer_activation))
                else:
                    self.add(keras.layers.Masking(mask_value=0))
                    self.add(keras.layers.Dense(hidden_layer_widths[0], activation=hidden_layer_activation))
            else:
                if input_size is None:
                    self.add(keras.layers.Dense(hidden_layer_widths[0], activation=hidden_layer_activation))
                else:
                    self.add(keras.layers.Dense(hidden_layer_widths[0], input_shape=(input_size,), activation=hidden_layer_activation))
            if dropout_rate is not None: 
                self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            for width in hidden_layer_widths[1:]:
                self.add (keras.layers.Dense(width,activation= hidden_layer_activation))
                if dropout_rate is not None: 
                    self.add (keras.layers.Dropout (dropout_rate, seed = 13))
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        else: raise Exception ("CustomNN: widths and depths are set incorrectly.\n Correct cases: (widths is int and depth is int) OR (widths is list and depth is None)")
        
        # APPLYING REGULARIZER: exlude the last Dense layer
        if regularizer == 'l2':
            for layer in self.layers[:-1]:
                layer.kernel_regularizer = tf.keras.regularizers.l2(l2=regularizer_weight)
                
        # INITIALIZER: 
        if initializer == 'he':
            initializer = keras.initializers.he_normal(seed = 13)
            for layer in self.layers: 
                layer.kernel_initializer = initializer
class Model1(keras.Model):
    def __init__(self, name = None,
                 group_nn_hidden_layer_widths = None, group_nn_hidden_layer_depth = None, 
                 technique_nn_hidden_layer_widths = None, technique_nn_hidden_layer_depth = None,
                 nn_output_size = None, config = None,
                 initializer = None, embeddings_initializer = "uniform" , dropout_rate = None, masking = None,
                 regularizer = None, regularizer_weight = None,
                 vocabs = None, limit_technique_features = None, limit_group_features = None,
                 *args, **kwargs):
        super().__init__(name = name, *args, **kwargs)
        
        if config != None:
            print ('---model built from config')
            group_nn_hidden_layer_widths = config['group_nn_hidden_layer_widths']
            group_nn_hidden_layer_depth = config['group_nn_hidden_layer_depth']
            technique_nn_hidden_layer_widths = config['technique_nn_hidden_layer_widths']
            technique_nn_hidden_layer_depth = config['technique_nn_hidden_layer_depth']
            nn_output_size = config['nn_output_size']
            regularizer = config['regularizer']
            regularizer_weight = config['regularizer_weight']
            if regularizer_weight != None: regularizer_weight = float (regularizer_weight)
            initializer = config['initializer']
            if config ['embeddings_initializer'] != None: embeddings_initializer = config['embeddings_initializer']
            dropout_rate = config['dropout_rate']
            if dropout_rate != None: dropout_rate = float (dropout_rate)
            masking = config['masking']
            
            limit_technique_features = config['limit_technique_features']
            limit_group_features = config['limit_group_features']

        ## 👉 input layers
        ### Group Inputs
        self.input_group_software_id =              tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_group_software_id')
        self.input_group_tactics =                  tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_group_tactics')
        self.input_group_interaction_rate =         tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.float32, name = 'input_group_interaction_rate')
        self.input_group_description =              tf.keras.layers.InputLayer(input_shape=(768,), dtype=tf.float32, name = 'input_group_description')
        
        ### Technique Inputs
        self.input_technique_data_sources =         tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_data_sources')
        # self.input_technique_defenses_bypassed =    tf.keras.Input(shape=(Noninput_e,), ragged=True, dtype= tf.string, name = 'input_technique_defenses_bypassed')
        self.input_technique_detection_name =       tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_detection_name')
        self.input_technique_mitigation_id =        tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_mitigation_id')
        # self.input_technique_permissions_required = tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_permissions_required')
        self.input_technique_platforms =            tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_platforms')
        self.input_technique_software_id =          tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_software_id')
        self.input_technique_tactics =              tf.keras.layers.InputLayer(input_shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_tactics')
        self.input_technique_interaction_rate =     tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.float32, name = 'input_technique_interaction_rate')
        self.input_technique_description =          tf.keras.layers.InputLayer(input_shape=(768,), dtype=tf.float32, name = 'input_technique_description')
        
        
        ## 👉 vectorization layers
        ### group and technique input shares two text vectorization layers: software and tactics
        self.vectorize_software_id =              tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_group_features['input_software_id'],split = None, vocabulary=vocabs ['input_software_id'], name = 'vectorize_software_id')
        self.vectorize_tactics =              tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_tactics']     ,split = None, vocabulary=vocabs ['input_tactics'], name = 'vectorize_tactics')
        # self.vectorize_software_id =          tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_software_id']         ,split = None, vocabulary=vocabs ['input_technique_software_id'], name = 'vectorize_technique_software_id')
        
        self.vectorize_technique_data_sources =         tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_data_sources']        ,split = None, vocabulary=vocabs ['input_technique_data_sources'], name = 'vectorize_technique_data_sources')
        # self.vectorize_technique_defenses_bypassed =    tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_defenses_bypassed']   ,split = None, vocabulary=vocabs ['input_technique_defenses_bypassed'], name = 'vectorize_technique_defenses_bypassed')
        self.vectorize_technique_detection_name =       tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_detection_name']      ,split = None, vocabulary=vocabs ['input_technique_detection_name'], name = 'vectorize_technique_detection_name')
        self.vectorize_technique_mitigation_id =        tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_mitigation_id']       ,split = None, vocabulary=vocabs ['input_technique_mitigation_id'], name = 'vectorize_technique_mitigation_id')
        # self.vectorize_technique_permissions_required = tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_permissions_required'],split = None, vocabulary=vocabs ['input_technique_permissions_required'], name = 'vectorize_technique_permissions_required')
        self.vectorize_technique_platforms =            tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_platforms']           ,split = None, vocabulary=vocabs ['input_technique_platforms'], name = 'vectorize_technique_platforms')
        
        ## 👉 embed layers
        ### group and technique input shares two embedding layers: software and tactics
        self.embed_software_id =              tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_software_id']), input_length= limit_group_features['input_software_id'],    output_dim=30, mask_zero= True, name = 'embed_software_id', embeddings_initializer= embeddings_initializer)
        self.embed_tactics =                tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_tactics']),       input_length= limit_technique_features['input_tactics'],    output_dim=5, mask_zero= True, name = 'embed_tactics', embeddings_initializer= embeddings_initializer)
        
        self.embed_technique_data_sources =         tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_data_sources']),        input_length= limit_technique_features['input_technique_data_sources'],         output_dim=10, mask_zero= True, name = 'embed_technique_data_sources', embeddings_initializer= embeddings_initializer)
        self.embed_technique_detection_name =       tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_detection_name']),      input_length= limit_technique_features['input_technique_detection_name'],       output_dim=10, mask_zero= True, name = 'embed_technique_detection_name', embeddings_initializer= embeddings_initializer)
        self.embed_technique_mitigation_id =        tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_mitigation_id']),       input_length= limit_technique_features['input_technique_mitigation_id'],        output_dim=10, mask_zero= True, name = 'embed_technique_mitigation_id', embeddings_initializer= embeddings_initializer)
        self.embed_technique_platforms =            tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_platforms']),           input_length= limit_technique_features['input_technique_platforms'],            output_dim=5, mask_zero= True, name = 'embed_technique_platforms', embeddings_initializer= embeddings_initializer)
        
        ## 👉 concatenate layer
        self.concatenate = tf.keras.layers.Concatenate (axis=1)
        
        # 👉 FNNS
        self.Group_NN = customNN_2(
            # input_size =  group_input_size,
                                 output_size = nn_output_size,
                                 hidden_layer_widths = group_nn_hidden_layer_widths,
                                 hidden_layer_depth =group_nn_hidden_layer_depth,
                                 initializer= initializer,
                                 name = 'Group_NN', 
                                 regularizer= regularizer, regularizer_weight= regularizer_weight, dropout_rate= dropout_rate,
                                 masking= masking)
        self.Technique_NN = customNN_2(
            # input_size = technique_input_size,
                                 output_size = nn_output_size,
                                 hidden_layer_widths = technique_nn_hidden_layer_widths,
                                 hidden_layer_depth = technique_nn_hidden_layer_depth,
                                 initializer= initializer,
                                 name = 'Technique_NN', 
                                 regularizer=regularizer, regularizer_weight= regularizer_weight, dropout_rate= dropout_rate,
                                 masking = masking)
         
        self.dot_product = keras.layers.Dot(axes= 1)
    
    def call(self, inputs):
        # 👉 input
        ## Group input
        self.input_group_software_id = inputs ['input_group_software_id']
        self.input_group_tactics = inputs ['input_group_tactics']
        self.input_group_interaction_rate = inputs['input_group_interaction_rate']
        self.input_group_description = inputs['input_group_description']
        ## Technique input
        self.input_technique_data_sources = inputs['input_technique_data_sources']
        # self.input_technique_defenses_bypassed = inputs['input_technique_defenses_bypassed']
        self.input_technique_detection_name = inputs['input_technique_detection_name']
        self.input_technique_mitigation_id = inputs['input_technique_mitigation_id']
        # self.input_technique_permissions_required = inputs['input_technique_permissions_required']
        self.input_technique_platforms = inputs['input_technique_platforms']
        self.input_technique_software_id = inputs['input_technique_software_id']
        self.input_technique_tactics = inputs['input_technique_tactics']
        self.input_technique_description = inputs['input_technique_description']
        self.input_technique_interaction_rate = inputs['input_technique_interaction_rate']
        
        # 👉 vectorization
        group_software_id = self.vectorize_software_id (self.input_group_software_id)
        group_tatics = self.vectorize_tactics (self.input_group_tactics)
        
        technique_data_sources = self.vectorize_technique_data_sources (self.input_technique_data_sources)
        # technique_defenses_bypassed = self.vectorize_technique_defenses_bypassed (self.input_technique_defenses_bypassed)
        technique_detection_name = self.vectorize_technique_detection_name (self.input_technique_detection_name)
        technique_mitigation_id = self.vectorize_technique_mitigation_id (self.input_technique_mitigation_id)
        # technique_permissions_required = self.vectorize_technique_permissions_required (self.input_technique_permissions_required)
        technique_platforms = self.vectorize_technique_platforms (self.input_technique_platforms)
        technique_software_id = self.vectorize_software_id (self.input_technique_software_id)
        technique_tactics = self.vectorize_tactics (self.input_technique_tactics)
        # 👉 embed
        group_software_id = self.embed_software_id (group_software_id)
        group_tatics = self.embed_tactics (group_tatics)
        
        technique_data_sources = self.embed_technique_data_sources (technique_data_sources)
        # technique_defenses_bypassed = self.embed_technique_defenses_bypassed (technique_defenses_bypassed)
        technique_detection_name = self.embed_technique_detection_name (technique_detection_name)
        technique_mitigation_id = self.embed_technique_mitigation_id (technique_mitigation_id)
        # technique_permissions_required = self.embed_technique_permissions_required (technique_permissions_required)
        technique_platforms = self.embed_technique_platforms (technique_platforms)
        technique_software_id = self.embed_software_id (technique_software_id)
        technique_tactics = self.embed_tactics (technique_tactics)

        # 👉 element-wise averaging
        group_software_id = tf.reduce_mean (group_software_id, axis = 1)
        group_tatics = tf.reduce_mean (group_tatics, axis = 1)
        
        technique_data_sources = tf.reduce_mean (technique_data_sources, axis = 1)
        technique_detection_name = tf.reduce_mean (technique_detection_name, axis = 1)
        technique_mitigation_id = tf.reduce_mean (technique_mitigation_id, axis = 1)
        technique_platforms = tf.reduce_mean (technique_platforms, axis = 1)
        technique_software_id = tf.reduce_mean (technique_software_id, axis = 1)
        technique_tactics = tf.reduce_mean (technique_tactics, axis = 1)
        
        # 👉 concatenate 
        group_concat = self.concatenate([
            self.input_group_interaction_rate,
            self.input_group_description,
            group_software_id,
            group_tatics,
        ])
        
        technique_concat = self.concatenate ([
            self.input_technique_interaction_rate,
            self.input_technique_description,
            technique_data_sources,
            # technique_defenses_bypassed,
            technique_detection_name,
            technique_mitigation_id,
            # technique_permissions_required,
            technique_platforms,
            technique_software_id,
            technique_tactics, 
        ])
        
        # 👉 FNNs
        output_Group = self.Group_NN(group_concat)
        output_Technique = self.Technique_NN(technique_concat)
        
        norm_output_Group = tf.linalg.l2_normalize (output_Group, axis = 1)
        norm_output_Technique = tf.linalg.l2_normalize (output_Technique, axis = 1)
        
        dot_product = self.dot_product ([norm_output_Group, norm_output_Technique])
        return dot_product

    def get_config (self):
        config = super(Model1, self).get_config()
        config.update ({
            "Group_NN": self.Group_NN,
            "Technique_NN": self.Technique_NN
        })
        return config 