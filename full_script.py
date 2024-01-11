# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle

# %%
import tensorflow as tf
# tensor flow functions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Dropout,GaussianNoise,RandomContrast
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# %%
# define path to image folders
img_folders_path = '256_ObjectCategories/sim_images/'

# %%
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# %%
# load data
datagen = ImageDataGenerator(rescale=1./255, # converts pixels in range 0,255 to between 0 and 1
                             # this will make every image contribute more evenly to the total loss
                            validation_split=0.2)

# if want to implement gray scale:
def to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

# then use preprocessing_function=to_grayscale in ImageDataGenerator
traingen_g = ImageDataGenerator(rescale=1./255,
                                validation_split = 0.2,
                                preprocessing_function=to_grayscale)

# if split into train and test dirs, could apply brightness range to
# only test set using "brightness_range" option in ImageDataGenerator
traingen_b = ImageDataGenerator(rescale=1./255,
                                validation_split = 0.2,
                                brightness_range = [0.5,1.5])

train_generator = datagen.flow_from_directory(
    img_folders_path,
    target_size=(150, 150),
    batch_size=32,
    shuffle = True,
    class_mode='categorical',
    subset='training'
)

# in case I want to experiment with brightness as an augmentation
# train_generator_b = traingen_b.flow_from_directory(
#     img_folders_path,
#     target_size=(150, 150),
#     batch_size=32,
#     shuffle = True,
#     class_mode='categorical',
#     subset='training'
# )

# or with gray scale as augmentation
# train_generator_g = traingen_g.flow_from_directory(
#     img_folders_path,
#     target_size=(150, 150),
#     batch_size=32,
#     shuffle = True,
#     class_mode='categorical',
#     subset='training'
# )


# will use this validation set as a test set, as val metrics can be used for early stopping and ability for tuning,
# but do not affect training (according to search I made)
test_generator = datagen.flow_from_directory(
    img_folders_path,
    target_size=(150, 150),
    batch_size=32,
    shuffle = True,
    class_mode='categorical',
    subset='validation',
)

# %%
# define important parameters
num_classes = len(np.unique(train_generator.classes))

# %%
# create early stopping
es = EarlyStopping(monitor='loss',patience=3,mode='min')
# could also add "start_from_epoch" specification in this es object
# in .fit() add the following to implement early stopping:
# callbacks=[es]

# %%
model_ffn = Sequential([
    RandomContrast(0, input_shape=(150, 150, 3)),
    GaussianNoise(0),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    # GaussianNoise(noise_sd),
    Dense(256, activation='relu'),
    Dropout(0.2),
    # GaussianNoise(noise_sd),
    Dense(128, activation='relu'),
    Dropout(0.1),
    # 5 classes: so get 5 output nuerons with softmax activiation function (to give probability of each class)
    Dense(num_classes, activation='softmax')
])
# there may also be some overfitting, so using some dropout could be helpful

model_ffn.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_ffn.fit(train_generator, epochs=10,
            #   steps_per_epoch=10,
                callbacks=[es])
model_ffn.summary()


# %%
model_cnn = Sequential([
    # syntax: Conv2D(num_filters, kernel_size (shape1,shape2), activation, input_shape for first layer)
    # RandomContrast(0.2, input_shape=(150, 150, 3)),
    # GaussianNoise(noise_sd),
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    # GaussianNoise(noise_sd),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # GaussianNoise(noise_sd),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # after convolutional layers, flatten the output, then use 1-2(/3?) dense layers
    Flatten(),
    # GaussianNoise(noise_sd),
    Dense(512, activation='relu'),
    Dropout(0.4),
    # GaussianNoise(noise_sd),
    Dense(256, activation='tanh'),
    Dropout(0.2),
    # GaussianNoise(noise_sd),
    # Dense(128, activation='relu'),
    # Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

model_cnn.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_cnn.fit(train_generator, epochs=10,
              callbacks = [es])
# default is to take len(train_generator steps per epoch)
model_cnn.summary()

# %%
# # save and restore models
# model_ffn.save("ffn_model")
# model_ffn_nc.save("ffn_model_nc")
# model_ffn_c.save("ffn_model_c")
# model_ffn_n.save("ffn_model_n")

# model_cnn.save("cnn_model")
# model_cnn_nc.save("cnn_model_nc")
# model_cnn_c.save("cnn_model_c")
# model_cnn_n.save("cnn_model_n")

# # reload the model
# model_ffn_reload = tf.keras.models.load_model('ffn_model')
# model_ffn_nc_reload = tf.keras.models.load_model('ffn_model_nc')
# model_cnn_reload = tf.keras.models.load_model('cnn_model')
# saver = tf.train.Saver(max_to_keep=1)
# with tf.Session() as sess:
#     # train your model, then:
#     savePath = saver.save(sess, 'my_ffn.ckpt')
# # To restore:

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('my_ffn.ckpt.meta')
#     saver.restore(sess, pathModel + 'someDir/my_model.ckpt')
#     # access a variable from the saved Graph, and so on:
#     someVar = sess.run('varName:0')

# %%
# results = model_ffn_reload.predict(test_generator)


# %%
# test out functionality to
n_sim = 2
test_accs = [np.nan]*n_sim

# %%
# evaluate the model
# _, train_acc = model_ffn.evaluate(train_generator, verbose=0)
_, test_accs[j] = model_ffn_reload.evaluate(test_generator)
_, test_accs[1] = model_ffn_nc_reload.evaluate(test_generator)
print(test_accs)

# %%
pred_digits = np.argmax(results,axis=1)
test_labels = test_generator.classes
len(test_labels)
len(pred_digits)

# %%
# combine model fits and accuracy assessments into one function
def network_sim(train_generator,test_generator,n_epochs,noise_sd=0,contrast_factor=0):
    """Fit Feed Forward and Convolutional Neural Networks with 2 factors: Random Noise and Random Contrast
    parameters:
        train_generator
        test_generator
        noise_sd: float [0.0,1.0], default = 0
            sd of Gaussian(0,sd) random noise to be added during training
        contrast_factor: float [0.0,1.0], defauly = 0
            factor to scale random contrast adjustment of images; 
            factor applied using following formula: 
            layer computes the mean of the image pixels in the channel and then adjusts each component x
            of each pixel to (x - mean) * contrast_factor + mean
    returns: *dictionary* {FFN_accuracy,CNN_accuracy}
    """
    # get number of classes in training data
    num_classes = len(np.unique(train_generator.classes))
    # define early stopping criteria
    es = EarlyStopping(monitor='loss',patience=3,mode='min')

    # feed forward nn fit
    model_ffn_nc = Sequential([
        RandomContrast(factor=contrast_factor, input_shape=(150, 150, 3)),
        GaussianNoise(noise_sd),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        # GaussianNoise(noise_sd),
        Dense(256, activation='relu'),
        Dropout(0.1),
        # GaussianNoise(noise_sd),
        Dense(128, activation='relu'),
        Dropout(0.1),
        # 5 classes: so get 5 output nuerons with softmax activiation function (to give probability of each class)
        Dense(num_classes, activation='softmax')
    ])
    # there may also be some overfitting, so using some dropout could be helpful

    model_ffn_nc.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print(f'Fitting feed forward network with Gaussian(0,{noise_sd}) noise and {contrast_factor} random contrast')
    model_ffn_nc.fit(train_generator, epochs=n_epochs,
                #   steps_per_epoch=10,
                    callbacks=[es])
    

    # cnn fit
    model_cnn_nc = Sequential([
        # syntax: Conv2D(num_filters, kernel_size (shape1,shape2), activation, input_shape for first layer)
        # recall: adding dropout can be helpful to remedy overfitting
        RandomContrast(factor=contrast_factor, input_shape=(150, 150, 3)),
        GaussianNoise(noise_sd),
        Conv2D(32, (3, 3), activation='relu'), #input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        # Dropout(0.1),
        # GaussianNoise(noise_sd),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # Dropout(0.1),
        # GaussianNoise(noise_sd),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # Dropout(0.1),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # Dropout(0.1),
        # after convolutional layers, flatten the output, then use 1-2(/3?) dense layers
        Flatten(),
        # GaussianNoise(noise_sd),
        Dense(512, activation='relu'),
        Dropout(0.2),
        # GaussianNoise(noise_sd),
        Dense(256, activation='tanh'),
        Dropout(0.2),
        # GaussianNoise(noise_sd),
        # Dense(128, activation='relu'),
        # Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])

    model_cnn_nc.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    print(f'Fitting CNN with Gaussian(0,{noise_sd}) noise and {contrast_factor} random contrast')
    model_cnn_nc.fit(train_generator, epochs=n_epochs,
                    callbacks = [es])
    # default is to take len(train_generator steps per epoch)

    # accuracy evaluations
    _, test_acc_ffn = model_ffn_nc.evaluate(test_generator)
    _, test_acc_cnn = model_cnn_nc.evaluate(test_generator)

    return {"FFN_accuracy": test_acc_ffn, "CNN_accuracy": test_acc_cnn}

# %%
# test network_sim
baseline_results = network_sim(train_generator,test_generator,n_epochs=1)
nc_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0.2,contrast_factor=0.1)
n_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0.2,contrast_factor=0)
c_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0,contrast_factor=0.1)


# %%
# test list structure before aplying in loop
n_sim = 3
ffn_base_accs = [np.nan]*n_sim
ffn_nc_accs = [np.nan]*n_sim
ffn_n_accs = [np.nan]*n_sim
ffn_c_accs = [np.nan]*n_sim
cnn_base_accs = [np.nan]*n_sim
cnn_nc_accs = [np.nan]*n_sim
cnn_n_accs = [np.nan]*n_sim
cnn_c_accs = [np.nan]*n_sim

ffn_base_accs[0] = baseline_results["FFN_accuracy"]
ffn_nc_accs[0] = nc_results["FFN_accuracy"]
ffn_n_accs[0] = n_results["FFN_accuracy"]
ffn_c_accs[0] = c_results["FFN_accuracy"]

cnn_base_accs[0] = baseline_results["CNN_accuracy"]
cnn_nc_accs[0] = nc_results["CNN_accuracy"]
cnn_n_accs[0] = n_results["CNN_accuracy"]
cnn_c_accs[0] = c_results["CNN_accuracy"]

# %%
# repeat process for simulation study
# then can also decide if we want to up noise/contrast or change filter

n_sims = 20
# initialize result lists
ffn_base_accs = [np.nan]*n_sim
ffn_nc_accs = [np.nan]*n_sim
ffn_n_accs = [np.nan]*n_sim
ffn_c_accs = [np.nan]*n_sim
cnn_base_accs = [np.nan]*n_sim
cnn_nc_accs = [np.nan]*n_sim
cnn_n_accs = [np.nan]*n_sim
cnn_c_accs = [np.nan]*n_sim
for j in range(n_sims):
    print(f"run {j+1}")
    # train networks
    baseline_results = network_sim(train_generator,test_generator,n_epochs=1)
    nc_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0.2,contrast_factor=0.1)
    n_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0.2,contrast_factor=0)
    c_results = network_sim(train_generator,test_generator,n_epochs=1,noise_sd=0,contrast_factor=0.1)

    # export results
    ffn_base_accs[j] = baseline_results["FFN_accuracy"]
    ffn_nc_accs[j] = nc_results["FFN_accuracy"]
    ffn_n_accs[j] = n_results["FFN_accuracy"]
    ffn_c_accs[j] = c_results["FFN_accuracy"]

    cnn_base_accs[j] = baseline_results["CNN_accuracy"]
    cnn_nc_accs[j] = nc_results["CNN_accuracy"]
    cnn_n_accs[j] = n_results["CNN_accuracy"]
    cnn_c_accs[j] = c_results["CNN_accuracy"]

# %%
# save results as pickles 
# ('wb': 'write byte/binary' => write using byte data: in binary mode)
# ffn
fileObj = open('ffn_base_accs.obj', 'wb')
pickle.dump(ffn_base_accs, fileObj)
fileObj.close()

fileObj = open('ffn_nc_accs.obj', 'wb')
pickle.dump(ffn_nc_accs, fileObj)
fileObj.close()

fileObj = open('ffn_n_accs.obj', 'wb')
pickle.dump(ffn_n_accs, fileObj)
fileObj.close()

fileObj = open('ffn_c_accs.obj', 'wb')
pickle.dump(ffn_c_accs, fileObj)
fileObj.close()

# cnn
fileObj = open('cnn_base_accs.obj', 'wb')
pickle.dump(ffn_base_accs, fileObj)
fileObj.close()

fileObj = open('cnn_nc_accs.obj', 'wb')
pickle.dump(ffn_nc_accs, fileObj)
fileObj.close()

fileObj = open('cnn_n_accs.obj', 'wb')
pickle.dump(ffn_n_accs, fileObj)
fileObj.close()

fileObj = open('cnn_c_accs.obj', 'wb')
pickle.dump(ffn_c_accs, fileObj)
fileObj.close()

# %%
# deserialize (reload) objects
fileObj = open('./shortsim2/fn_base_accs.obj', 'rb')
ffn_base_accs_reload2 = pickle.load(fileObj)
fileObj.close()
print(ffn_base_accs_reload2)

fileObj = open('./shortsim2/ffn_c_accs.obj', 'rb')
ffn_c_accs_reload2 = pickle.load(fileObj)
fileObj.close()

fileObj = open('./shortsim2/fn_n_accs.obj', 'rb')
ffn_n_accs_reload2 = pickle.load(fileObj)
fileObj.close()

fileObj = open('./shortsim2/ffn_nc_accs.obj', 'rb')
ffn_nc_accs_reload2 = pickle.load(fileObj)
fileObj.close()

fileObj = open('./shortsim2/cnn_base_accs.obj', 'rb')
cnn_base_accs_reload2 = pickle.load(fileObj)
fileObj.close()
print(cnn_base_accs_reload2)

fileObj = open('./shortsim2/cnn_c_accs.obj', 'rb')
cnn_c_accs_reload2 = pickle.load(fileObj)
fileObj.close()

fileObj = open('./shortsim2/cnn_n_accs.obj', 'rb')
cnn_n_accs_reload2 = pickle.load(fileObj)
fileObj.close()

fileObj = open('./shortsim2/cnn_nc_accs.obj', 'rb')
cnn_nc_accs_reload2 = pickle.load(fileObj)
fileObj.close()


print("ANN BASE accuracy:", np.mean(ffn_base_accs_reload2), "with sd:", np.std(ffn_base_accs_reload2))
print("CNN BASE accuracy:", np.mean(cnn_base_accs_reload2), "with sd:", np.std(cnn_base_accs_reload2))
print("ANN NOISY accuracy:", np.mean(ffn_n_accs_reload2), "with sd:", np.std(ffn_n_accs_reload2))
print("CNN NOISY accuracy:", np.mean(cnn_n_accs_reload2), "with sd:", np.std(cnn_n_accs_reload2))
print("ANN GRAY accuracy:", np.mean(ffn_c_accs_reload2), "with sd:", np.std(ffn_c_accs_reload2))
print("CNN GRAY accuracy:", np.mean(cnn_c_accs_reload2), "with sd:", np.std(cnn_c_accs_reload2))
print("ANN NOISY GRAY accuracy:", np.mean(ffn_nc_accs_reload2), "with sd:", np.std(ffn_nc_accs_reload2))
print("CNN NOISY GRAY accuracy:", np.mean(cnn_nc_accs_reload2), "with sd:", np.std(cnn_nc_accs_reload2))

# %%
# deserialize (reload) objects
# original sim: 12 epochs, 20 sims, to gray scale as in to 2d, rather than b&w
fileObj = open('./pickles_shortsim/ffn_base_accs.obj', 'rb')
ffn_base_accs_reload = pickle.load(fileObj)
fileObj.close()
print(ffn_base_accs_reload)

fileObj = open('./pickles_shortsim/ffn_c_accs.obj', 'rb')
ffn_c_accs_reload = pickle.load(fileObj)
fileObj.close()

fileObj = open('./pickles_shortsim/ffn_n_accs.obj', 'rb')
ffn_n_accs_reload = pickle.load(fileObj)
fileObj.close()

fileObj = open('./pickles_shortsim/ffn_nc_accs.obj', 'rb')
ffn_nc_accs_reload = pickle.load(fileObj)
fileObj.close()

fileObj = open('./pickles_shortsim/cnn_base_accs.obj', 'rb')
cnn_base_accs_reload = pickle.load(fileObj)
fileObj.close()
print(cnn_base_accs_reload)

fileObj = open('./pickles_shortsim/cnn_c_accs.obj', 'rb')
cnn_c_accs_reload = pickle.load(fileObj)
fileObj.close()

fileObj = open('./pickles_shortsim/cnn_n_accs.obj', 'rb')
cnn_n_accs_reload = pickle.load(fileObj)
fileObj.close()

fileObj = open('./pickles_shortsim/cnn_nc_accs.obj', 'rb')
cnn_nc_accs_reload = pickle.load(fileObj)
fileObj.close()


print("ANN BASE accuracy:", np.mean(ffn_base_accs_reload), "with sd:", np.std(ffn_base_accs_reload))
print("CNN BASE accuracy:", np.mean(cnn_base_accs_reload), "with sd:", np.std(cnn_base_accs_reload))
print("ANN NOISY accuracy:", np.mean(ffn_n_accs_reload), "with sd:", np.std(ffn_n_accs_reload))
print("CNN NOISY accuracy:", np.mean(cnn_n_accs_reload), "with sd:", np.std(cnn_n_accs_reload))
print("ANN GRAY accuracy:", np.mean(ffn_c_accs_reload), "with sd:", np.std(ffn_c_accs_reload))
print("CNN GRAY accuracy:", np.mean(cnn_c_accs_reload), "with sd:", np.std(cnn_c_accs_reload))
print("ANN NOISY GRAY accuracy:", np.mean(ffn_nc_accs_reload), "with sd:", np.std(ffn_nc_accs_reload))
print("CNN NOISY GRAY accuracy:", np.mean(cnn_nc_accs_reload), "with sd:", np.std(cnn_nc_accs_reload))

# %%
sim_results_dict = {"Method": ["ANN_BASE","ANN_NOISY","ANN_GRAY", "ANN_NOISY_GRAY","CNN_BASE","CNN_NOISY","CNN_GRAY", "CNN_NOISY_GRAY"],
                    "Mean Accuracy":[np.mean(ffn_base_accs_reload),np.mean(ffn_n_accs_reload),np.mean(ffn_c_accs_reload),np.mean(ffn_nc_accs_reload),
                                     np.mean(cnn_base_accs_reload),np.mean(cnn_n_accs_reload),np.mean(cnn_c_accs_reload),np.mean(cnn_nc_accs_reload)],
                    "SD Accuracy": [np.std(ffn_base_accs_reload),np.std(ffn_n_accs_reload),np.std(ffn_c_accs_reload),np.std(ffn_nc_accs_reload),
                                    np.std(cnn_base_accs_reload),np.std(cnn_n_accs_reload),np.std(cnn_c_accs_reload),np.std(cnn_nc_accs_reload)]}
sim_res_df = pd.DataFrame(sim_results_dict)
sim_res_df

# %%
# output table to latex
sim_res_df.to_latex

# %%
# deserialize (reload) objects
# corrected sim: 12 epochs, 20 sims, fully to gray scale, back as 3d rgb image
fileObj = open('./sim_gray_corrected/ffn_base_accs.obj', 'rb')
ffn_base_accs_reloadc = pickle.load(fileObj)
fileObj.close()
print(ffn_base_accs_reloadc)

fileObj = open('./sim_gray_corrected/ffn_c_accs.obj', 'rb')
ffn_c_accs_reloadc = pickle.load(fileObj)
fileObj.close()

fileObj = open('./sim_gray_corrected/ffn_n_accs.obj', 'rb')
ffn_n_accs_reloadc = pickle.load(fileObj)
fileObj.close()

fileObj = open('./sim_gray_corrected/ffn_nc_accs.obj', 'rb')
ffn_nc_accs_reloadc = pickle.load(fileObj)
fileObj.close()

fileObj = open('./sim_gray_corrected/cnn_base_accs.obj', 'rb')
cnn_base_accs_reloadc = pickle.load(fileObj)
fileObj.close()
print(cnn_base_accs_reloadc)

fileObj = open('./sim_gray_corrected/cnn_c_accs.obj', 'rb')
cnn_c_accs_reloadc = pickle.load(fileObj)
fileObj.close()

fileObj = open('./sim_gray_corrected/cnn_n_accs.obj', 'rb')
cnn_n_accs_reloadc = pickle.load(fileObj)
fileObj.close()

fileObj = open('./sim_gray_corrected/cnn_nc_accs.obj', 'rb')
cnn_nc_accs_reloadc = pickle.load(fileObj)
fileObj.close()


print("ANN BASE accuracy:", np.mean(ffn_base_accs_reloadc), "with sd:", np.std(ffn_base_accs_reloadc))
print("CNN BASE accuracy:", np.mean(cnn_base_accs_reloadc), "with sd:", np.std(cnn_base_accs_reloadc))
print("ANN NOISY accuracy:", np.mean(ffn_n_accs_reloadc), "with sd:", np.std(ffn_n_accs_reloadc))
print("CNN NOISY accuracy:", np.mean(cnn_n_accs_reloadc), "with sd:", np.std(cnn_n_accs_reloadc))
print("ANN GRAY accuracy:", np.mean(ffn_c_accs_reloadc), "with sd:", np.std(ffn_c_accs_reloadc))
print("CNN GRAY accuracy:", np.mean(cnn_c_accs_reloadc), "with sd:", np.std(cnn_c_accs_reloadc))
print("ANN NOISY GRAY accuracy:", np.mean(ffn_nc_accs_reloadc), "with sd:", np.std(ffn_nc_accs_reloadc))
print("CNN NOISY GRAY accuracy:", np.mean(cnn_nc_accs_reloadc), "with sd:", np.std(cnn_nc_accs_reloadc))

# %%
sim_results_dict = {"Method": ["DNN BASE","DNN NOISY","DNN GRAY", "DNN NOISY+GRAY","CNN BASE","CNN NOISY","CNN GRAY", "CNN NOISY+GRAY"],
                    "Mean Accuracy":[np.mean(ffn_base_accs_reloadc),np.mean(ffn_n_accs_reloadc),np.mean(ffn_c_accs_reloadc),np.mean(ffn_nc_accs_reloadc),
                                     np.mean(cnn_base_accs_reloadc),np.mean(cnn_n_accs_reloadc),np.mean(cnn_c_accs_reloadc),np.mean(cnn_nc_accs_reloadc)],
                    "SD": [np.std(ffn_base_accs_reloadc),np.std(ffn_n_accs_reloadc),np.std(ffn_c_accs_reloadc),np.std(ffn_nc_accs_reloadc),
                                    np.std(cnn_base_accs_reloadc),np.std(cnn_n_accs_reloadc),np.std(cnn_c_accs_reloadc),np.std(cnn_nc_accs_reloadc)]}
sim_res_dfc = pd.DataFrame(sim_results_dict)
sim_res_dfc

# output table to latex
print(sim_res_dfc.style.to_latex(position = 't'))

# %%
# attempt to look at confusion matrix/classification report
# predictions appear to be in shuffled order,
# so performance not matching what is shown when model.evaluate() is run

# test_labels = test_generator.classes_
# pred_digits = np.argmax(model_cnn.predict(test_generator),axis=1)
# from sklearn.metrics import classification_report, confusion_matrix
# confusion_matrix(y_true=test_labels,y_pred=pred_digits)
# print(classification_report(test_labels,pred_digits))

# real data example
img_folders_path = '256_ObjectCategories/sim_images'

# %%
# format data for training
# load data
datagen = ImageDataGenerator(rescale=1./255, # converts pixels in range 0,255 to between 0 and 1
                             # this will make every image contribute more evenly to the total loss
                            validation_split=0.2)

train_generator = datagen.flow_from_directory(
    img_folders_path,
    target_size=(150, 150),
    batch_size=32,
    # shuffle = True,
    class_mode='categorical',
    subset='training'
)


# will use this validation set as a test set, as val metrics can be used for early stopping and ability for tuning,
# but do not affect training (according to search I made)
val_generator = datagen.flow_from_directory(
    img_folders_path,
    target_size=(150, 150),
    batch_size=32,
    # shuffle = True,
    class_mode='categorical',
    subset='validation',
)

# cnn fit
model_cnn = Sequential([
    # syntax: Conv2D(num_filters, kernel_size (shape1,shape2), activation, input_shape for first layer)
    # recall: adding dropout can be helpful to remedy overfitting
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    #Dropout(0.1),
    # GaussianNoise(noise_sd),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Dropout(0.1),
    # GaussianNoise(noise_sd),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Dropout(0.1),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Dropout(0.1),
    # after convolutional layers, flatten the output, then use 1-2(/3?) dense layers
    Flatten(),
    # GaussianNoise(noise_sd),
    Dense(512, activation='relu'),
    Dropout(0.4),
    # GaussianNoise(noise_sd),
    Dense(256, activation='relu'),
    Dropout(0.2),
    # GaussianNoise(noise_sd),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

# model_cnn_nc.compile(optimizer='SGD',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

model_cnn.compile(optimizer=Adam(learning_rate = 0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
n_epochs = 50
model_cnn.fit(train_generator, validation_data=val_generator, epochs=n_epochs,
                 callbacks = [es])


# save model history
losses = model_cnn.history.history

fileObj = open('cnn_losses_real.obj', 'wb')
pickle.dump(losses, fileObj)
fileObj.close()

n_epochs = 40
model_cnn.fit(train_generator, validation_data=val_generator, epochs=n_epochs,
                 callbacks = [es])

# save model history
losses = model_cnn.history.history

fileObj = open('losses_cnn.obj', 'wb')
pickle.dump(losses, fileObj)
fileObj.close()


# fit model from pretraining VGG16
from keras.applications import VGG16

# Load the VGG16 base model without the top classification layer
base_model_pt = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# download link finishes with "_notop"

# Freeze all layers in the base model to retain their learned weights
for layer in base_model_pt.layers:
    layer.trainable = False

# Optionally, unfreeze the top N layers
# N = 5
# for layer in base_model_pt.layers[-N:]:
#     layer.trainable = True

# Add custom layers on top
x = base_model_pt.output
x = Flatten()(x) # try both Flatten and GlobalAveragePooling2D
# x = GlobalAveragePooling2D(2,2)
x = Dense(512, activation='relu')(x)
x= Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model_transfer = Model(inputs=base_model_pt.input, outputs=predictions)

# Compile with a smaller learning rate
model_transfer.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

model_transfer.fit(train_generator, validation_data=val_generator, epochs=12,
                   )

# save model object
model_transfer.save("vgg16_transfer")

# save model history
t_losses = model_transfer.history.history

fileObj = open('model_t_losses.obj', 'wb')
pickle.dump(t_losses, fileObj)
fileObj.close()

# get preds and save pred digits
# for val data
labels = val_generator.classes
# print(labels)
res = model_transfer.predict(val_generator)
# print(res)
pred_digits = np.argmax(res,axis=1)

# save pred digits
fileObj = open('model_t_digits.obj', 'wb')
pickle.dump(pred_digits, fileObj)
fileObj.close()

# reload model object
cnnObj = open('./real_data_eg/model_cnn.obj','rb')
model_cnn_reload = pickle.load(cnnObj)
cnnObj.close()

# %%
# reload model performance (losses and accuracies from 50 epoch run on server)
lossesObj = open('./real_data_eg/cnn_losses_real.obj','rb')
model_cnn_losses_reload = pickle.load(lossesObj)
lossesObj.close()

# 40 epoch run
lossesObj = open('./real_data_eg/losses_cnn.obj','rb')
model_cnn_losses_reload2 = pickle.load(lossesObj)
lossesObj.close()

# reload losses for VGG16 run
lossesObj = open('./real_data_eg/model_t_losses.obj','rb')
model_cnn_losses_reloadt = pickle.load(lossesObj)
lossesObj.close()


# %%
print(model_cnn_losses_reload)
print(model_cnn_losses_reload2)
print(model_cnn_losses_reloadt)

# %%
# plot model performance
# evaluation
# example code
losses_long = model_cnn_losses_reload
# print(losses_long)
df_loss_l = pd.DataFrame(losses_long)
display(df_loss_l.sort_values(by='val_accuracy',ascending=False).head())

df_loss_l.plot(
    figsize=(8, 6), grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc=(1.01,0))  # extra code
 # extra code
plt.show()
# plt.savefig("cnn_learning_curves_plot.png") 

# %%
# plot model performance
# evaluation
# example code
losses_long2 = model_cnn_losses_reload2
# print(losses_long)
df_loss_l2 = pd.DataFrame(losses_long2)
display(df_loss_l2.sort_values(by='val_accuracy',ascending=False).head())

df_loss_l2.plot(
    figsize=(8, 6), grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc=(1.01,0))  # extra code
 # extra code
plt.show()
# plt.savefig("cnn_learning_curves_plot.png") 

# %%
# plot model performance
# evaluation
# example code
losses_t = model_cnn_losses_reloadt
# print(losses_long)
df_loss_t = pd.DataFrame(losses_t)
display(df_loss_t.sort_values(by='val_accuracy',ascending=False).head())

df_loss_t.plot(
    figsize=(8, 6), grid=True, xlabel="Epoch",
    title='VGG16 learning curves',
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc=(1.01,0))  # extra code
plt.show()


# %%
# # reload model to get predictions and classification report
# # this still had issues, appears predictions in shuffled order
# # from my cnn
# fileObj = open('real_data_eg/cnn_val_digits.obj','rb')
# mycnn_digits = pickle.load(fileObj)
# fileObj.close()

# # from VGG16 model transfer
# fileObj = open('real_data_eg/model_t_digits.obj','rb')
# t_digits = pickle.load(fileObj)
# fileObj.close()

# # %%
# # get classification report
# # test_labels = val_generator.classes
# # pred_digits = np.argmax(model_cnn_nc.predict(val_generator),axis=1)
# # for all data
# labels =val_generator.classes
# # print(labels)
# res = model_ffn_nc.predict(all_dat_generator)
# # print(res)
# pred_digits = np.argmax(res,axis=1)
# # print(pred_digits)
# print(confusion_matrix(y_true=labels,y_pred=pred_digits))
# print(classification_report(labels,pred_digits))

# # %%
# # cnn
# labels = val_generator.classes
# # print(labels)
# # res = model_cnn_nc.predict(all_dat_generator)
# # print(res)
# # pred_digits = np.argmax(res,axis=1)
# # print(pred_digits)
# print(confusion_matrix(y_true=labels,y_pred=t_digits))
# print(classification_report(labels,t_digits,))





