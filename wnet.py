from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input,  Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from unet import generate_unet_model

K.set_image_data_format('channels_first')

def generate_wnet_model(wparams):
    input_channels = wparams['input_channels']
    output_channels = wparams['output_channels']
    scale = wparams['scale']
    patch_shape = wparams['patch_shape']
    loss_weights = wparams['loss_weights']

    f1 = decoder_maker(input_channels, patch_shape, output_channels, scale[0], use_skip_connections=True)
    f2 = decoder_maker(input_channels, patch_shape, output_channels, scale[0], use_skip_connections=True)
    f3 = decoder_maker(input_channels, patch_shape, output_channels, scale[0], use_skip_connections=True)
    f4 = decoder_maker(output_channels, patch_shape, output_channels, scale[1], use_skip_connections=True)
    
    input_vol = Input(shape=(1, ) + patch_shape)
    input_prob_1 = Input(shape=(1, ) + patch_shape)
    input_prob_2 = Input(shape=(1, ) + patch_shape)
    input_prob_3 = Input(shape=(1, ) + patch_shape)

    f1_out = f1(Concatenate(axis=1)([input_vol, input_prob_1]))
    f2_out = f2(Concatenate(axis=1)([input_vol, input_prob_2]))
    f3_out = f3(Concatenate(axis=1)([input_vol, input_prob_3]))
    
    ccat = Add()([f1_out, f2_out, f3_out])
    
    f_out = f4(ccat)
    
    f = Model(inputs=[input_vol, input_prob_1, input_prob_2, input_prob_3], outputs=[f1_out, f2_out, f3_out, ccat, f_out])

    def mae_loss(y_true, y_pred):
        mask = K.batch_flatten(K.cast(K.not_equal(y_true, 0), 'float32'))
        return mae(y_true, y_pred) * mask

    loss = [mae_loss, mae_loss, mae_loss, mae_loss, mae_loss]
    f.compile(optimizer=Adam(), loss=loss, loss_weights=loss_weights)

    return f

def decoder_maker(input_channels, patch_shape, output_channels, scale, use_skip_connections=True):
    input_shape = (input_channels, ) + patch_shape
    fc_layer_filters = output_channels
    use_batchnorm = False

    inp, pred = generate_unet_model(input_shape, fc_layer_filters, scale, use_batchnorm)
    return Model(inputs=[inp], outputs=[pred])

def mae(y_true, y_pred):
    return K.abs(K.batch_flatten(y_pred - y_true))
