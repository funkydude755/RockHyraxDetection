from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

#wierd configs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


PATH = "/home/ok/OAI/Bunnys"
#get folders
folders = [f for f in listdir(PATH) if isdir(join(PATH, f))]
model = VGG16()
bunny_pictures = []
non_bunny = []
for folder in folders:
    folder_path = join(PATH, folder)
    #non bunny data in the folder
    non_bunny_pics = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    # for pic in non_bunny_pics:

        # image = Image.open(pic)
        # image = image.resize((224,224))
        # print(image.format)
        # print(image.mode)
        # print(image.size)


    # print(folder_path)
    # print (onlyfiles)
    non_bunny += non_bunny_pics
    # print(len(non_bunny))
    #bunny data inside folder/bunnys
    bunny_path = join(folder_path, "bunnys")
    # print(bunny_path)
    bunnyfiles = [join(bunny_path, f) for f in listdir(bunny_path) if isfile(join(bunny_path, f))]



    # print(bunnyfiles)
    bunny_pictures += bunnyfiles
    for pic in bunny_pictures:
        image = load_img(pic, target_size=(224, 224))
        # image.show()
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        label = decode_predictions(yhat)
        label = label[0][0]
        print('%s (%.2f%%)' % (label[1], label[2] * 100))
        break
    # print(len(bunny_pictures))
verdict = [1 for i in bunny_pictures]
# print(len(verdict))
verdict += [0 for i in non_bunny]
# print(len(verdict))
pictures = bunny_pictures + non_bunny
# print(len(pic))

data = pd.DataFrame()
data['pictures'] = pictures
data['has_bunny'] = verdict
data.to_csv(join(PATH,'bunny_data.csv'))