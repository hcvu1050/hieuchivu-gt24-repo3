"""
Last update: 2023-10-29
V0.5: each feature has its own embedding layer
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
    def __init__(self, input_sizes=None, name = None,
                 group_nn_hidden_layer_widths = None, group_nn_hidden_layer_depth = None, 
                 technique_nn_hidden_layer_widths = None, technique_nn_hidden_layer_depth = None,
                 nn_output_size = None, config = None,
                 initializer = None, dropout_rate = None, masking = None,
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
            dropout_rate = config['dropout_rate']
            if dropout_rate != None: dropout_rate = float (dropout_rate)
            masking = config['masking']
            
            limit_technique_features = config['limit_technique_features']
            limit_group_features = config['limit_group_features']
            
        # group_input_size = input_sizes['group_feature_size']
        # technique_input_size = input_sizes['technique_feature_size']
        
        # self.input_Group = keras.layers.Input (shape= (group_input_size,), name = 'input_Group')
        # self.input_Technique = keras.layers.Input (shape= (technique_input_size,), name = 'input_Technique')
        
        #  each feature has its own vectorization-embedding layer
        ## 👉 input layers
        self.input_group_software_id =              tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_group_software_id')
        self.input_technique_data_sources =         tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_data_sources')
        self.input_technique_defenses_bypassed =    tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_defenses_bypassed')
        self.input_technique_detection_name =       tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_detection_name')
        self.input_technique_mitigation_id =        tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_mitigation_id')
        self.input_technique_permissions_required = tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_permissions_required')
        self.input_technique_platforms =            tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_platforms')
        self.input_technique_software_id =          tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_software_id')
        self.input_technique_tactics =              tf.keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_tactics')
        
        ## 👉 vectorization layers
        self.vectorize_group_software_id =              tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_group_features['input_group_software_id'],split = None, vocabulary=vocabs ['input_group_software_id'], name = 'vectorize_group_software_id')
        self.vectorize_technique_data_sources =         tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_data_sources']        ,split = None, vocabulary=vocabs ['input_technique_data_sources'], name = 'vectorize_technique_data_sources')
        self.vectorize_technique_defenses_bypassed =    tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_defenses_bypassed']   ,split = None, vocabulary=vocabs ['input_technique_defenses_bypassed'], name = 'vectorize_technique_defenses_bypassed')
        self.vectorize_technique_detection_name =       tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_detection_name']      ,split = None, vocabulary=vocabs ['input_technique_detection_name'], name = 'vectorize_technique_detection_name')
        self.vectorize_technique_mitigation_id =        tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_mitigation_id']       ,split = None, vocabulary=vocabs ['input_technique_mitigation_id'], name = 'vectorize_technique_mitigation_id')
        self.vectorize_technique_permissions_required = tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_permissions_required'],split = None, vocabulary=vocabs ['input_technique_permissions_required'], name = 'vectorize_technique_permissions_required')
        self.vectorize_technique_platforms =            tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_platforms']           ,split = None, vocabulary=vocabs ['input_technique_platforms'], name = 'vectorize_technique_platforms')
        self.vectorize_technique_software_id =          tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_software_id']         ,split = None, vocabulary=vocabs ['input_technique_software_id'], name = 'vectorize_technique_software_id')
        self.vectorize_technique_tactics =              tf.keras.layers.TextVectorization (max_tokens=1000, output_mode= 'int', output_sequence_length= limit_technique_features['input_technique_tactics']             ,split = None, vocabulary=vocabs ['input_technique_tactics'], name = 'vectorize_technique_tactics')
        
        ## 👉 embed layers
        # self.embed_group_software_id =              tf.keras.layers.Embedding (input_dim=463+2, input_length=46 , output_dim=21, mask_zero= True, name = 'embed_group_software_id')
        # self.embed_technique_data_sources =         tf.keras.layers.Embedding (input_dim=105+2, input_length=14 , output_dim=10, mask_zero= True, name = 'embed_technique_data_sources')
        # self.embed_technique_defenses_bypassed =    tf.keras.layers.Embedding (input_dim=24+2 , input_length=8  , output_dim=5, mask_zero= True, name = 'embed_technique_defenses_bypassed')
        # self.embed_technique_detection_name =       tf.keras.layers.Embedding (input_dim=105+2, input_length=14 , output_dim=10, mask_zero= True, name = 'embed_technique_detection_name')
        # self.embed_technique_mitigation_id =        tf.keras.layers.Embedding (input_dim=43+2 , input_length=11 , output_dim=7, mask_zero= True, name = 'embed_technique_mitigation_id')
        # self.embed_technique_permissions_required = tf.keras.layers.Embedding (input_dim=5+2  , input_length=4  , output_dim=3, mask_zero= True, name = 'embed_technique_permissions_required')
        # self.embed_technique_platforms =            tf.keras.layers.Embedding (input_dim=11+2 , input_length=10 , output_dim=4, mask_zero= True, name = 'embed_technique_platforms')
        # self.embed_technique_software_id =          tf.keras.layers.Embedding (input_dim=635+2, input_length=334, output_dim=25, mask_zero= True, name = 'embed_technique_software_id')
        # self.embed_technique_tactics =              tf.keras.layers.Embedding (input_dim=14+2 , input_length=4  , output_dim=4, mask_zero= True, name = 'embed_technique_tactics')
        
        self.embed_group_software_id =              tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_group_software_id']),             input_length= limit_group_features['input_group_software_id'],                  output_dim=20, mask_zero= True, name = 'embed_group_software_id')
        self.embed_technique_data_sources =         tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_data_sources']),        input_length= limit_technique_features['input_technique_data_sources'],         output_dim=10, mask_zero= True, name = 'embed_technique_data_sources')
        self.embed_technique_defenses_bypassed =    tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_defenses_bypassed']),   input_length= limit_technique_features['input_technique_defenses_bypassed'],    output_dim=5, mask_zero= True, name = 'embed_technique_defenses_bypassed')
        self.embed_technique_detection_name =       tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_detection_name']),      input_length= limit_technique_features['input_technique_detection_name'],       output_dim=10, mask_zero= True, name = 'embed_technique_detection_name')
        self.embed_technique_mitigation_id =        tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_mitigation_id']),       input_length= limit_technique_features['input_technique_mitigation_id'],        output_dim=10, mask_zero= True, name = 'embed_technique_mitigation_id')
        self.embed_technique_permissions_required = tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_permissions_required']),input_length= limit_technique_features['input_technique_permissions_required'], output_dim=5, mask_zero= True, name = 'embed_technique_permissions_required')
        self.embed_technique_platforms =            tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_platforms']),           input_length= limit_technique_features['input_technique_platforms'],            output_dim=5, mask_zero= True, name = 'embed_technique_platforms')
        self.embed_technique_software_id =          tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_software_id']),         input_length= limit_technique_features['input_technique_software_id'],          output_dim=20, mask_zero= True, name = 'embed_technique_software_id')
        self.embed_technique_tactics =              tf.keras.layers.Embedding (input_dim = 2 + len(vocabs ['input_technique_tactics']),             input_length= limit_technique_features['input_technique_tactics'],              output_dim=5, mask_zero= True, name = 'embed_technique_tactics')
        
        ## 👉 flatten layer
        self.flatten = tf.keras.layers.Flatten ()
        
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
        self.input_group_software_id = inputs ['input_group_software_id']
        self.input_technique_data_sources = inputs['input_technique_data_sources']
        self.input_technique_defenses_bypassed = inputs['input_technique_defenses_bypassed']
        self.input_technique_detection_name = inputs['input_technique_detection_name']
        self.input_technique_mitigation_id = inputs['input_technique_mitigation_id']
        self.input_technique_permissions_required = inputs['input_technique_permissions_required']
        self.input_technique_platforms = inputs['input_technique_platforms']
        self.input_technique_software_id = inputs['input_technique_software_id']
        self.input_technique_tactics = inputs['input_technique_tactics']
        # 👉 vectorization
        group_software_id = self.vectorize_group_software_id (self.input_group_software_id)
        technique_data_sources = self.vectorize_technique_data_sources (self.input_technique_data_sources)
        technique_defenses_bypassed = self.vectorize_technique_defenses_bypassed (self.input_technique_defenses_bypassed)
        technique_detection_name = self.vectorize_technique_detection_name (self.input_technique_detection_name)
        technique_mitigation_id = self.vectorize_technique_mitigation_id (self.input_technique_mitigation_id)
        technique_permissions_required = self.vectorize_technique_permissions_required (self.input_technique_permissions_required)
        technique_platforms = self.vectorize_technique_platforms (self.input_technique_platforms)
        technique_software_id = self.vectorize_technique_software_id (self.input_technique_software_id)
        technique_tactics = self.vectorize_technique_tactics (self.input_technique_tactics)
        # 👉 embed
        group_software_id = self.embed_group_software_id (group_software_id)
        technique_data_sources = self.embed_technique_data_sources (technique_data_sources)
        technique_defenses_bypassed = self.embed_technique_defenses_bypassed (technique_defenses_bypassed)
        technique_detection_name = self.embed_technique_detection_name (technique_detection_name)
        technique_mitigation_id = self.embed_technique_mitigation_id (technique_mitigation_id)
        technique_permissions_required = self.embed_technique_permissions_required (technique_permissions_required)
        technique_platforms = self.embed_technique_platforms (technique_platforms)
        technique_software_id = self.embed_technique_software_id (technique_software_id)
        technique_tactics = self.embed_technique_tactics (technique_tactics)
        # 👉 flatten
        group_software_id = self.flatten(group_software_id)
        technique_data_sources = self.flatten(technique_data_sources)
        technique_defenses_bypassed = self.flatten(technique_defenses_bypassed)
        technique_detection_name = self.flatten(technique_detection_name)
        technique_mitigation_id = self.flatten(technique_mitigation_id)
        technique_permissions_required = self.flatten(technique_permissions_required)
        technique_platforms = self.flatten(technique_platforms)
        technique_software_id = self.flatten(technique_software_id)
        technique_tactics = self.flatten(technique_tactics)
        # 👉 concatenate 
        technique_concat = self.concatenate ([
            technique_data_sources,
            technique_defenses_bypassed,
            technique_detection_name,
            technique_mitigation_id,
            technique_permissions_required,
            technique_platforms,
            technique_software_id,
            technique_tactics
        ])
        
        # 👉 FNNs
        output_Group = self.Group_NN(group_software_id)
        output_Technique = self.Technique_NN(technique_concat)
        
        norm_output_Group = tf.linalg.l2_normalize (output_Group, axis = 1)
        norm_output_Technique = tf.linalg.l2_normalize (output_Technique, axis = 1)
        
        dot_product = self.dot_product ([norm_output_Group, norm_output_Technique])
        return dot_product
