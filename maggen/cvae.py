import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization,Flatten, Dense
from tensorflow.keras.layers import Reshape, Conv1DTranspose, Activation, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau as RLROP
from tensorflow.keras.callbacks import LearningRateScheduler as LRS



tf.compat.v1.disable_eager_execution()


class VAE :
    def __init__(self,input_shape,latent_dim) :

        self.input_shape = input_shape
        self.latent_dim = latent_dim  

        self.encoder = None
        self.target_learning_branch_1 = None
        self.target_learning_branch_2 = None
        self.decoder = None
        self.model = None
        
        self._shape_before_bottleneck = None
        self._model_input = None
        self._decoder_output = None
        self._build()

    #=========================================="ENCODER"========================================
    def _build_encoder(self) :
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input,bottleneck,name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape)

    def _add_conv_layers(self,x):
        # conv layer 1
        x = Conv1D(64, 5, 2, padding='SAME')(x)  # num of filter = 64, kernel size = 3, stride = 1
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        # conv layer 2
        x = Conv1D(128, 5, 2, padding='SAME')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        # conv layer 3
        x = Conv1D(256, 5, 2,padding='SAME')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        # conv layer 4
        x = Conv1D(256, 3, 1,padding='SAME')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        return x

    def _add_bottleneck(self,x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(1024, activation='sigmoid')(x)
        self.z_mean = Dense(self.latent_dim,activation = 'linear')(x)
        self.z_log_var = Dense(self.latent_dim,activation = 'linear')(x)
    
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=K.shape(self.z_mean),mean=0,stddev=1)
            sampled_point = z_mean+K.exp(z_log_var/2)*epsilon
            return sampled_point
    
        # Reparameterization
        z = Lambda(sampling, output_shape=(self.latent_dim))([self.z_mean, self.z_log_var])
        return z

    #==================================="Target-Learning Branch-1"===========================================

    def _build_target_learning_branch_1(self):
       branch_input = self._add_branch_input_1()
       dense_layer = self._add_target_dense_layer_1(branch_input)
       self.branch_output = self._add_branch_output_1(dense_layer)
       self.target_learning_branch_1 = Model(branch_input,self.branch_output,name="target-learning-branch-1")

    def _add_branch_input_1(self):
       return Input(shape=self.latent_dim)

    def _add_target_dense_layer_1(self,branch_input):
       x = Dense(128,activation="relu")(branch_input)
       x = Dense(32,activation="relu")(x)
       return x

    def _add_branch_output_1(self,x):
       x = Dense(1,activation="sigmoid")(x)
       return x

  #==================================="Target-Learning Branch-2"===========================================

    def _build_target_learning_branch_2(self):
       branch_input = self._add_branch_input_2()
       dense_layer = self._add_target_dense_layer_2(branch_input)
       self.branch_output = self._add_branch_output_2(dense_layer)
       self.target_learning_branch_2 = Model(branch_input,self.branch_output,name="target-learning-branch-2")

    def _add_branch_input_2(self):
       return Input(shape=self.latent_dim)

    def _add_target_dense_layer_2(self,branch_input):
       x = Dense(128,activation="relu")(branch_input)
       x = Dense(32,activation="relu")(x)
       return x

    def _add_branch_output_2(self,x):
       x = Dense(1,activation="sigmoid")(x)
       return x


    #======================================"DECODER"============================================
    def _build_decoder(self) :
       decoder_input = self._add_decoder_input()
       dense_layer = self._add_dense_layer(decoder_input)
       reshape_layer = self._add_reshape_layer(dense_layer)
       conv_transpose_layers = self._add_conv_transpose_layer(reshape_layer)
       self._decoder_output = self._add_decoder_output(conv_transpose_layers)
       self.decoder = Model(decoder_input,self._decoder_output,name="decoder")

    def _add_decoder_input(self):
       return Input(shape=self.latent_dim)
 
    def _add_dense_layer(self,decoder_input):
       x = Dense(1024,activation="relu")(decoder_input)
       num_neurons = np.prod(self._shape_before_bottleneck)
       x = Dense(num_neurons)(x)   
       return x

    def _add_reshape_layer(self,dense_layer):
       x =  Reshape(self._shape_before_bottleneck)(dense_layer)
       return x

    def _add_conv_transpose_layer(self,x) :
       # layer 4
       x = BatchNormalization()(x)
       x = Conv1DTranspose(256, 3, 1, padding='SAME', activation="relu")(x)
       # layer 3
       x = BatchNormalization()(x)
       x = Conv1DTranspose(256, 5, 2, padding='SAME', activation="relu")(x)
       # layer 2
       x = BatchNormalization()(x)
       x = Conv1DTranspose(128, 5, 2, padding='SAME', activation="relu")(x)
       # layer 1
       x = BatchNormalization()(x)
       x = Conv1DTranspose(64, 5, 2, padding='SAME', activation="relu")(x)
       return x

    def _add_decoder_output(self,x):
        x = Conv1D(filters=3,kernel_size=1,strides=1,padding='SAME')(x)
        output_layer = Activation("sigmoid")(x)
        return output_layer


    #========================================"Build the model"=============================================    
    def _build(self):
       self._build_encoder()
       self._build_decoder()
       self._build_target_learning_branch_1()
       self._build_target_learning_branch_2()
       self._build_vae()

    def _build_vae(self):
       model_input = self._model_input
       y_hat_1 = self.target_learning_branch_1(self.encoder(model_input)) #output from target branch
       y_hat_2 = self.target_learning_branch_2(self.encoder(model_input)) #output from target branch
       self.recons_output = self.decoder(self.encoder(model_input)) # output from decoder
       self.model = Model(inputs=model_input,outputs=[y_hat_1,y_hat_2,self.recons_output],name="vae")
       

    def summary(self):
       self.encoder.summary()
       self.decoder.summary()
       self.target_learning_branch_1.summary()
       self.target_learning_branch_2.summary()
       self.model.summary()

    #======================================"Loss Function"===============================================

    def recons_loss(self,y_target,y_predicted):
        error = y_target - y_predicted    
        mse = K.mean(K.square(error), axis=[1,2])
        return mse

    def KL_loss(self,y_target,y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=1)
        return kl_loss
    
    # property loss for hform regression
    def property_loss_1(self,y_target,y_predicted):
       error = y_target - y_predicted        
       mse = K.mean(K.square(error), axis=[1])
       return 2*mse

    # property loss for magnetization regression
    def property_loss_2(self,y_target,y_predicted):
       error = y_target - y_predicted
       mse = K.mean(K.square(error), axis=[1])
       return 2*mse

    def combined_loss(self,y_target,y_predicted) :
        reconstruction_loss = self.recons_loss(y_target,y_predicted)
        kl_loss = self.KL_loss(y_target,y_predicted)
        combined_loss = reconstruction_loss + 1e-6*kl_loss
        return combined_loss

    #========================================"Compile & Train"===========================================
    
    # learning rate scheduler
    def scheduler(self,epoch,lr):
        if epoch < 100 :
            return lr
        if epoch >= 100 and epoch < 200 :
            return 1e-4
        if epoch >= 200 :
            return 5e-5

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss={"target-learning-branch-1":self.property_loss_1,
                                 "target-learning-branch-2":self.property_loss_2,
                                 "decoder":self.combined_loss})

    def train(self, train_X, test_X, train_y_1, test_y_1, train_y_2, test_y_2, batch_size, num_epochs):
        x_train = train_X
        y_train = (train_y_1, train_y_2,train_X)
        x_test = test_X
        y_test = (test_y_1,test_y_2,test_X)
        schedule_lr = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        reduce_lr = RLROP(monitor='loss', factor=0.3, patience=4, min_lr=1e-6)
        history = self.model.fit(x_train,
                                 y_train,
                                 validation_data=(x_test,y_test),
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 callbacks= [reduce_lr,schedule_lr],
                                 shuffle=True)    
        # save the history to plot loss curve
        epochs = list(np.arange(1,num_epochs+1))
        train_loss = history.history["loss"]
        test_loss = history.history["val_loss"]
        data_dict = {"epochs":epochs,"train_loss":train_loss, "test_loss":test_loss}
        df = pd.DataFrame(data_dict)
        df.to_csv("loss-vs-epochs.csv",index=False)


    #====================================="Reconstruction"================================================
    def reconstruct(self,images):
        latent_rep = self.encoder.predict(images)
        y_hat_1 = self.target_learning_branch_1.predict(latent_rep) #output from target learning branch
        y_hat_2 = self.target_learning_branch_2.predict(latent_rep) #output from target branch
        recons_images = self.decoder.predict(latent_rep)
        return latent_rep, y_hat_1, y_hat_2, recons_images


    def getLatentRepresentation(self,images):
        latent_rep = self.encoder.predict(images)
        return latent_rep

    def getPredictedHform(self,latent_rep):
        y_hat = self.target_learning_branch_1.predict(latent_rep)
        return y_hat
   
    def getPredictedMagnetization(self,latent_rep):
        y_hat = self.target_learning_branch_2.predict(latent_rep)
        return y_hat

    def getReconstructedImage(self,latent_rep):
        recons_images = self.decoder.predict(latent_rep)
        return recons_images



    #==============================="Save & Load the model"===============================================
    
    def save(self,save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self,weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls,save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path,"rb") as f :
            parameters = pickle.load(f)
        vae = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        vae.load_weights(weights_path)
        return vae

    def _create_folder_if_it_doesnt_exist(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self,save_folder):
        parameters = [self.input_shape, self.latent_dim]
        save_path = os.path.join(save_folder,"parameters.pkl")
        with open(save_path,"wb") as f :
            pickle.dump(parameters,f)

    def _save_weights(self,save_folder):
        save_path = os.path.join(save_folder,"weights.h5")
        self.model.save_weights(save_path)        



if __name__ == "__main__" :
    vae = VAE(input_shape=(144,3),
              latent_dim=256)
    print(vae.summary())






