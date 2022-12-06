import numpy as np
import cv2 
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.utils import load_img, img_to_array
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.datasets import load_files   
from tqdm import tqdm
from .consts import *

class DogBreeder:
    def __init__(self):
        self.dog_model = None
    
    def load_model_from_file(self):
        self.dog_model = load_model(MODEL_PATH)
        
    def predict_dog(self, image_path):
        img = self.path_to_tensor(image_path)
        feature = self.extract_Resnet50(img)
        if not self.dog_model:
            self.load_model_from_file()
        return np.argmax(self.dog_model.predict(feature), axis=1)[0]
    
    def get_dog_name(self, breed_num):
        return DOG_NAMES.get(breed_num, '')
    
    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    
    def extract_Resnet50(self, tensor):
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
        
    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 
    
    def ResNet50_predict_labels(self, img_path):
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.path_to_tensor(img_path))
        ResNet50_model = ResNet50(weights='imagenet')
        return np.argmax(ResNet50_model.predict(img))
    
    def paths_to_tensor(self, img_paths):
        def path_to_tensor_safe(img_path):
            try:
                return self.path_to_tensor(img_path)
            except:
                print('error parsing image {}'.format(img_path))
                return None
        return [tensor for img_path in tqdm(img_paths) for tensor in [path_to_tensor_safe(img_path)] if tensor is not None]

    def human_face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('cnn/haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0
    
    def create_model(self):
        # load train, test, and validation datasets
        train_files, train_targets = self.load_dataset('cnn/dogImages/train')
        valid_files, valid_targets = self.load_dataset('cnn/dogImages/valid')
        test_files, test_targets = self.load_dataset('cnn/dogImages/test')
        
        train_tensors = self.paths_to_tensor(train_files)
        valid_tensors = self.paths_to_tensor(valid_files)
        test_tensors = self.paths_to_tensor(test_files)
        
        train_resnet = np.vstack([self.extract_Resnet50(feature) for feature in train_tensors])
        valid_resnet = np.vstack([self.extract_Resnet50(feature) for feature in valid_tensors])
        test_resnet = np.vstack([self.extract_Resnet50(feature) for feature in test_tensors])

        resnet_model = Sequential()
        resnet_model.add(GlobalAveragePooling2D(input_shape=train_resnet.shape[1:]))
        resnet_model.add(Dropout(0.3))
        resnet_model.add(Dense(266, activation='relu'))
        resnet_model.add(Dropout(0.1))
        resnet_model.add(Dense(133, activation='softmax'))

        resnet_model.summary()
        resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=MODEL_PATH, verbose=1, save_best_only=True, save_weights_only=False)
        
        datagen_resnet_train = ImageDataGenerator(
            width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
            height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
            horizontal_flip=True) # randomly flip images horizontally

        # create and configure augmented image generator
        datagen_resnet_valid = ImageDataGenerator(
            width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
            height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
            horizontal_flip=True) # randomly flip images horizontally

        # fit augmented image generator train_resnet data
        datagen_resnet_train.fit(train_resnet)
        datagen_resnet_valid.fit(valid_resnet)

        epochs = 1
        batch_size = 20
        resnet_model.fit_generator(datagen_resnet_train.flow(train_resnet, train_targets, batch_size=batch_size),
                            steps_per_epoch=train_resnet.shape[0] // batch_size,
                            epochs=epochs, verbose=1, callbacks=[checkpointer],
                            validation_data=datagen_resnet_valid.flow(valid_resnet, valid_targets, batch_size=batch_size),
                            validation_steps=valid_resnet.shape[0] // batch_size)
        
        resnet_predictions = [np.argmax(resnet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_resnet]

        resnet_model.load_weights(MODEL_PATH) 
        # report test accuracy
        test_accuracy = 100*np.sum(np.array(resnet_predictions)==np.argmax(test_targets, axis=1))/len(resnet_predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
        
    def load_dataset(self, path):
        data = load_files(path)
        dog_files = [filename for filename in np.array(data['filenames']) if filename.endswith('.jpg')]
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets

        
        

# def lambda_handler(event, context):
#     is_dog = dog_detector(img_path)
#     if is_dog:
#         is_human = False
#     else:
#         is_human = face_detector(img_path)
    
#     if not (is_dog or is_human):
#         print('Neither human nor dog detected!')
    
#     breed_num = predict_dog(img_path).split('.')[-1]
#     breed_name = DOG_NAMES[breed_num]
    
#     return {
#         'statusCode': 200,
#         'body': json.dumps(breed_name)
#     }


