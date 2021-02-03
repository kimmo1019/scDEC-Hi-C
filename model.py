import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
#the default is relu function
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)
    #return tf.maximum(0.0, x)
    #return tf.nn.tanh(x)
    #return tf.nn.elu(x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2] ,tf.shape(y)[3]])], 3)

class Autoencoder(object):
    def __init__(self,input_dim, name,hidden_dim=50,layer_norm=False):
        self.input_dim = input_dim
        self.name = name
        self.hidden_dim = hidden_dim
        self.layer_norm = layer_norm

    def __call__(self, x ,reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            #(None,49,49,1)
            conv = tcl.convolution2d(x, 128, [3,3], [1,1],activation_fn=tf.nn.relu,padding='SAME')
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            conv = tcl.max_pool2d(conv,[2,2])
            #(None,24,24,128)
            print(conv)
            conv = tcl.convolution2d(conv, 128, [3,3], [1,1],activation_fn=tf.nn.relu,padding='SAME')
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            conv = tcl.max_pool2d(conv,[2,2])
            #(None,12,12,128)
            print(conv)
            conv = tcl.convolution2d(conv, 64, [3,3], [1,1],activation_fn=tf.nn.relu,padding='SAME')
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            conv = tcl.max_pool2d(conv,[2,2])
            #(None,6,6,64)
            print(conv)
            fc = tcl.flatten(conv)
            print(fc)
            fc = tcl.fully_connected(fc, 500, activation_fn=tf.nn.relu)
            fc = tf.nn.dropout(fc,0.2)
            fc = tcl.fully_connected(fc, 50, activation_fn=tf.identity)
            encoded = fc
            fc = tcl.fully_connected(fc, 500, activation_fn=tf.nn.relu)
            fc = tf.nn.dropout(fc,0.2)
            fc = tcl.fully_connected(fc, 6*6*64, activation_fn=tf.nn.relu)
            bs = tf.shape(x)[0]
            conv = tf.reshape(fc, [bs, 6, 6, 64])
            #(None,6,6,64)
            print(conv)
            conv = tf.nn.conv2d_transpose(
                value = conv,
                filter = tf.get_variable(name="filter1", initializer=tf.random_normal_initializer(stddev=0.02),shape=(3,3,128,64)), 
                output_shape = [bs,12,12,128],
                strides = [1,2,2,1],
                padding='SAME'
            )
            conv = tf.nn.relu(conv)
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            #(None,12,12,128)
            print(conv)
            conv = tf.nn.conv2d_transpose(
                value = conv,
                filter = tf.get_variable(name="filter2", initializer=tf.random_normal_initializer(stddev=0.02),shape=(3,3,64,128)), 
                output_shape = [bs,24,24,64],
                strides = [1,2,2,1],
                padding='SAME'
            )
            conv = tf.nn.relu(conv)
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            #(None,24,24,64)
            print(conv)
            conv = tf.nn.conv2d_transpose(
                value = conv,
                filter = tf.get_variable(name="filter3", initializer=tf.random_normal_initializer(stddev=0.02),shape=(3,3,1,64)), 
                output_shape = [bs,49,49,1],
                strides = [1,2,2,1],
                padding='VALID'
            )
            logits = conv
            conv = tf.sigmoid(conv)
            if self.layer_norm:
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
            decoded = conv 
            print(conv)
            return encoded, decoded, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            #fc = tcl.batch_norm(fc)
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = tcl.batch_norm(fc)
                #fc = leaky_relu(fc)
                fc = tf.nn.tanh(fc)
            
            output = tcl.fully_connected(
                fc, 1, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, concat_every_fcl=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        
    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            y = z[:,self.input_dim:]
            fc = tcl.fully_connected(
                z, self.nb_units,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
            fc = leaky_relu(fc)
            #fc = tf.nn.dropout(fc,0.1)
            if self.concat_every_fcl:
                fc = tf.concat([fc, y], 1)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = True)
                
                fc = leaky_relu(fc)
                if self.concat_every_fcl:
                    fc = tf.concat([fc, y], 1)
            
            output = tcl.fully_connected(
                fc, self.output_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tcl.l2_regularizer(2.5e-5),
                #activation_fn=tf.sigmoid
                activation_fn=tf.identity
                )
            #output = tc.layers.batch_norm(output,decay=0.9,scale=True,updates_collections=None,is_training = True)
            output = tf.sigmoid(output)
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]



class Encoder(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tcl.fully_connected(
                x, self.nb_units,
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers-1):
                fc = tcl.fully_connected(
                    fc, self.nb_units,
                    #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                #weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )               
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            #return output[:, 0:self.feat_dim], y, logits
            return output, y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]



class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='scHiC_250'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]

            if self.dataset=="scHiC_250":
                z = tf.reshape(z, [bs, 250, 250, 23])
                conv = tcl.convolution2d(z, 64, [7,7],[5,5],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                #(bs, 50, 50, 64)
                #print(conv)
                conv = leaky_relu(conv)
                for _ in range(self.nb_layers-1):
                    conv = tcl.convolution2d(conv, 128, [7,7],[5,5],
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        activation_fn=tf.identity
                        )
                    #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                    conv = leaky_relu(conv)
                #(bs, 10, 10, 128)
                #print(conv)
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
                #(bs, 5, 5, 64)

            elif self.dataset=="scHiC_49":
                z = tf.reshape(z, [bs, 49, 49, 23])
                conv = tcl.convolution2d(z, 64, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #print('Dis',conv)
                #(bs, 25 25, 64)
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #(bs, 13, 13, 64)
                #print('Dis',conv)
                conv = tcl.convolution2d(conv, 32, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #(bs, 7, 7, 32)
                #print('Dis',conv)
            fc = tcl.flatten(conv)
            fc = tcl.fully_connected(
                fc, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='scHiC_250',is_training=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True, return_logits=False,):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            y = z[:,-self.nb_classes:]
            #yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = tf.concat([fc, y], 1)
            #change 5-->3
            fc = tcl.fully_connected(
                fc, 3*3*64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tf.reshape(fc, tf.stack([bs, 3, 3, 64]))

            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = conv_cond_concat(fc,yb)

            conv = tf.nn.conv2d_transpose(
                value = fc,
                filter = tf.get_variable(name="filter1", initializer=tf.random_normal_initializer(stddev=0.02),shape=(4,4,64,64)), 
                output_shape = [bs,6,6,64],
                strides = [1,2,2,1],
                padding='SAME'
            )
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = tf.nn.relu(conv)
            #(bs,6,6,64)
            #print(conv)
            if self.dataset=="scHiC_49":
                conv = tf.nn.conv2d_transpose(
                    value = conv,
                    filter = tf.get_variable(name="filter2", initializer=tf.random_normal_initializer(stddev=0.02),shape=(4,4,64,64)), 
                    output_shape = [bs,12,12,64],
                    strides = [1,2,2,1],
                    padding='SAME'
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
                #(bs,12,12,128)
                print('generator',conv)
                conv = tf.nn.conv2d_transpose(
                    value = conv,
                    filter = tf.get_variable(name="filter3", initializer=tf.random_normal_initializer(stddev=0.02),shape=(4,4,64,64)), 
                    output_shape = [bs,24,24,64],
                    strides = [1,2,2,1],
                    padding='SAME'
                )
                print('generator',conv)
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
                #(bs,24,24,128)
                conv = tf.nn.conv2d_transpose(
                    value = conv,
                    filter = tf.get_variable(name="filter4", initializer=tf.random_normal_initializer(stddev=0.02),shape=(4,4,23,64)), 
                    output_shape = [bs,48,48,23],
                    strides = [1,2,2,1],
                    padding='SAME'
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
                print('generator',conv)
                paddings = tf.constant([[0, 0], [1, 0],[1,0],[0,0]])
                conv = tf.pad(conv,paddings)
                #(bs,48,48,128)
                # conv = tf.nn.conv2d_transpose(
                #     value = conv,
                #     filter = tf.get_variable(name="filter5", initializer=tf.random_normal_initializer(stddev=0.02),shape=(4,4,23,128)), 
                #     output_shape = [bs,49,49,23],
                #     strides = [1,1,1,1],
                #     padding='VALID'
                # )
                # #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                # print('generator',conv)

            elif self.dataset=="scHiC_250":
                conv = tf.nn.conv2d_transpose(
                    value = conv,
                    filter = tf.get_variable(name="filter2", initializer=tf.random_normal_initializer(stddev=0.02),shape=(7,7,64,64)), 
                    output_shape = [bs,50,50,64],
                    strides = [1,5,5,1],
                    padding='SAME'
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
                #print(conv)
                #(bs,50,50,64)

                conv = tf.nn.conv2d_transpose(
                    value = conv,
                    filter = tf.get_variable(name="filter3", initializer=tf.random_normal_initializer(stddev=0.02),shape=(7,7,23,64)), 
                    output_shape = [bs,250,250,23],
                    strides = [1,5,5,1],
                    padding='SAME'
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                #(bs,250,250,23)
            print('generator',conv)
            logits = tf.reshape(conv, [bs, -1])
            output = tf.sigmoid(logits)
            if return_logits:
                return output, logits
            else:
                return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#encoder for images, H()
class Encoder_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='scHiC_250',cond=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset=="scHiC_250":
                x = tf.reshape(x, [bs, 250, 250, 23])
                conv = tcl.convolution2d(x,64,[7,7],[5,5],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
                #(bs,50,50,64)
                #print(conv)
                for _ in range(self.nb_layers-1):
                    conv = tcl.convolution2d(conv, self.nb_units, [7,7],[5,5],
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                        activation_fn=tf.identity
                        )
                    conv = leaky_relu(conv)
                #(bs,10,10,256)
                #print(conv)
                conv = tcl.convolution2d(conv,64,[4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
                #(bs,5,5,64)

            elif self.dataset=="scHiC_49":
                x = tf.reshape(x, [bs, 49, 49, 23])
                conv = tcl.convolution2d(x,64,[4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #(bs,25,25,64)
                #print('encoder',conv)
                conv = tcl.convolution2d(conv,64,[4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #(bs,13,13,64)
                #print('encoder',conv)
                conv = tcl.convolution2d(conv,32,[4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
                #(bs,7,7,32)
                #print('encoder',conv)

            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, 
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity)
            fc = leaky_relu(fc)

            output = tcl.fully_connected(
                fc, self.output_dim, 
                activation_fn=tf.identity
                )
            logits = output[:, -self.nb_classes:]
            y = tf.nn.softmax(logits)
            return output[:, :-self.nb_classes], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#####for pathology imgs#######

class Discriminator_pathology_img(object):
    def __init__(self, input_dim, name, nb_layers=2,nb_units=256,dataset='pathology'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            z = tf.reshape(z, [bs, 256, 256, 3])
            conv = tcl.convolution2d(z, 32, [4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            conv = tcl.max_pool2d(conv,[2,2])
            #(bs, 64, 64, 32)
            conv = leaky_relu(conv)
            for _ in range(2):
                conv = tcl.convolution2d(conv, 64, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.identity
                    )
                conv = tcl.max_pool2d(conv,[2,2])
                #conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None)
                conv = leaky_relu(conv)
            #(bs, 4, 4, 64)
            #fc = tf.reshape(conv, [bs, -1])
            fc = tcl.flatten(conv)
            #(bs, 1568)
            fc = tcl.fully_connected(
                fc, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
                )
            #fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None)
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, 1, 
                activation_fn=tf.identity
                )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


#generator for images, G()
class Generator_pathology_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',is_training=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        #with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:       
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            y = z[:,-10:]
            #yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            fc = tcl.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = tf.concat([fc, y], 1)
            fc = tcl.fully_connected(
                fc, 8*8*128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tc.layers.batch_norm(fc,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            fc = tf.nn.relu(fc)
            #fc = conv_cond_concat(fc,yb)
            conv = tcl.convolution2d_transpose(
                fc, 64, [4,4], [2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #(bs,16,16,64)
            conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
            conv = tf.nn.relu(conv)
            for _ in range(3):
                conv = tcl.convolution2d_transpose(
                    conv, 64, [4,4], [2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                )
                conv = tc.layers.batch_norm(conv,decay=0.9,scale=True,updates_collections=None,is_training = self.is_training)
                conv = tf.nn.relu(conv)
            #(bs,128,128,64)

            output = tcl.convolution2d_transpose(
                conv, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.nn.sigmoid
            )
            output = tf.reshape(output, [bs, -1])
            #(0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Encoder_pathology_img(object):
    def __init__(self, nb_classes, output_dim, name, nb_layers=2,nb_units=256,dataset='mnist',cond=True):
        self.nb_classes = nb_classes
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 256, 256, 3])
            conv = tcl.convolution2d(x,64,[4,4],[2,2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
                )
            conv = leaky_relu(conv)
            conv = tcl.max_pool2d(conv,[2,2])
            #(bs,64,64,64)
            for _ in range(self.nb_layers-1):
                conv = tcl.convolution2d(conv, self.nb_units, [4,4],[2,2],
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                    activation_fn=tf.identity
                    )
                conv = leaky_relu(conv)
            conv = tcl.max_pool2d(conv,[4,4],stride=4)
            #(bs,8,8,64)
            conv = tcl.flatten(conv)
            fc = tcl.fully_connected(conv, 1024, 
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity)
            
            fc = leaky_relu(fc)
            output = tcl.fully_connected(
                fc, self.output_dim, 
                activation_fn=tf.identity
                )        
            logits = output[:, -self.nb_classes:]
            y = tf.nn.softmax(logits)
            return output[:, :-self.nb_classes], y, logits        

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

if __name__=="__main__":
    y1 = tf.placeholder(tf.float32, [100, 49, 49, 1], name='y1')
    ae = Autoencoder('ae')
    a,b = ae(y1,reuse=False)
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    print(a,b)
