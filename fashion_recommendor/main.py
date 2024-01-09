import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalAvgPool2D()
])

# print(model.summary())

def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img)
    normalized_result=result/norm(result)
    return normalized_result

filename=[]

for file in os.listdir('images'):
    filename.append(os.path.join('images',file))

feature_list=[]

for file in tqdm(filename):
    feature_list.append(extract_features(file,model))

print(feature_list[0:5])

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filename,open('filenames.pkl','wb'))
