from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
from PIL import Image
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model

import matplotlib.pyplot as plt

import tensorflow as tf
import time

#wierd configs
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory
sess = tf.Session(config=config)

start = time.time()
PATH = "/home/ok/OAI/Bunnys/RockHyrexDetection/Classification/vgg_16_data"
#get folders
folders = [f for f in listdir(PATH) if isdir(join(PATH, f))]

# base_model = VGG16(weights='imagenet', include_top=False)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 450, 3))

new_model = base_model.output
new_model = GlobalAveragePooling2D()(new_model)
new_model = Dense(1024, activation='relu')(new_model) #we add dense layers so that the model can learn more complex functions and classify for better results.
new_model = Dense(1024, activation='relu')(new_model)
new_model = Dense(512, activation='relu')(new_model)
preds=Dense(2,activation='softmax')(new_model)
model=Model(inputs=base_model.input,outputs=preds)
# for layer in model.layers:
#     layer.trainable=False

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True
# for i,layer in enumerate(model.layers):
#   print(i,layer.name)


end = time.time()
print("model setup took: {}".format(str(end-start)))

start = time.time()
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest") #included in our dependencies

train_generator=train_datagen.flow_from_directory(PATH,
                                                 target_size=(300, 450),
                                                 color_mode='rgb',
                                                 batch_size=30,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)
end = time.time()
print("training took: {}".format(str(end-start)))
# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

model.save("HyrexDetector")

test_dir="/home/ok/Desktop/test"

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(300, 450),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)
# print(type(filenames))
print(filenames)

test_generator.reset()
start = time.time()
predict = model.predict_generator(test_generator,steps = nb_samples)
end = time.time()

print("test took: {}".format(str(end-start)))
# print(len(predict))
# print (predict)
predicted_class_indices=np.argmax(predict,axis=1)
# print(len(predicted_class_indices))
print(predicted_class_indices)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

lab0 = [x for x in range(len(filenames)) if filenames[x][:10]==labels[0]]
lab1 = [x for x in range(len(filenames)) if filenames[x][:13]==folders[1][2:]]
gt = np.zeros(nb_samples,dtype=int)
gt[lab1]=1
print("seccess: {0:.2f}%".format((sum(gt==predicted_class_indices)/len(predicted_class_indices))*100))





# plt.plot(predict.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(predict.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()