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
from PIL import Image
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
import seaborn as sns
from .consts import *

class DogBreeder:
    '''
    Used to create the model and predict images
    '''
    def __init__(self):
        self.dog_model = None
    
    def load_model_from_file(self):
        '''
        Load model from default MODEL_PATH file
        '''
        self.dog_model = load_model(MODEL_PATH)
        
    def predict_dog_from_path(self, image_path: str):
        '''
        Predicts dog breed
        
        input:
            image_path (str): path to image
            
        output:
            breed number (numpy.int64)
        '''
        tensor = self.path_to_tensor(image_path)
        return self.predict_from_tensor(tensor)
        
    def predict_dog(self, img_bytes: bytes):
        '''
        Predicts dog breed
        
        input:
            img_bytes (bytes): image in bytes format
            
        output:
            breed number (numpy.int64)
        '''
        tensor = self.bytes_to_tensor(img_bytes)
        return self.predict_from_tensor(tensor)
    
    def predict_from_tensor(self, tensor: np.array):
        '''
        Predicts dog breed
        
        input:
            tensor (numpy.array): image in tensor with shape (1, 224, 224, 3)
            
        output:
            breed number (numpy.int64)
        '''
        feature = self.extract_Resnet50(tensor)
        if not self.dog_model:
            self.load_model_from_file()
        return np.argmax(self.dog_model.predict(feature), axis=1)[0]
    
    def get_dog_name(self, breed_num: int):
        '''
        Returns breed name from breed num
        
        input:
            breed_num (numpy.int64): breed nuber
            
        output:
            breed_name (str)
        '''
        return DOG_NAMES[breed_num]
    
    def path_to_tensor(self, img_path: str):
        '''
        Convert image to 4D tensor with shape (1, 224, 224, 3)
        
        input:
            img_path (str): path to image
            
        output:
            tensor (numpy.array)
        '''
        
        # loads RGB image as PIL.Image.Image type
        img = load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    
    def bytes_to_tensor(self, img_bytes: bytes):
        '''
        Convert image to 4D tensor with shape (1, 224, 224, 3)
        
        input:
            img_bytes (bytes): image in bytes format
            
        output:
            tensor (numpy.array)
        '''
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.NEAREST)
        x = img_to_array(img)
        print(x)
        return np.expand_dims(x, axis=0)
    
    def extract_Resnet50(self, tensor: np.array):
        '''
        Uses ResNet50 model to predict features
        
        input:
            tensor (numpy.array): 4D tensor with shape (1, 224, 224, 3)
            
        output:

        '''
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
        
    def dog_detector_from_path(self, img_path: str):
        '''
        Predict dog breed number
        
        input:
            img_path (str): path to image
            
        output:
            breed_num (numpy.int64)
        '''
        tensor = preprocess_input(self.path_to_tensor(img_path))
        return self.dog_detector_from_tensor(tensor)
     
    def dog_detector(self, img_bytes: bytes):
        '''
        Predict dog breed number
        
        input:
            img_bytes (bytes): image in bytes format
            
        output:
            breed_num (numpy.int64)
        '''
        tensor = self.bytes_to_tensor(img_bytes)
        return self.dog_detector_from_tensor(tensor)
    
    def dog_detector_from_tensor(self, tensor: np.array):
        '''
        Predict dog breed number
        
        input:
            tensor (numpy.array): 4D tensor with shape (1, 224, 224, 3)
            
        output:
            breed_num (numpy.int64)
        '''
        prediction = self.ResNet50_predict_labels(tensor)
        return ((prediction <= 268) & (prediction >= 151)) 
    
    def ResNet50_predict_labels(self, tensor: np.array):
        '''
        Uses ResNet50 model to predict image type
        
        input:
            tensor (numpy.array): 4D tensor with shape (1, 224, 224, 3)
            
        output:
            image type number
        '''
        ResNet50_model = ResNet50(weights='imagenet')
        return np.argmax(ResNet50_model.predict(tensor))
    
    def paths_to_tensor(self, img_paths: str):
        '''
        Convert images in img_paths to 4D tensors with shape (1, 224, 224, 3)
        
        input:
            img_paths (str): path to folder containing images
            
        output:
            tensors (list[numpy.array])
        '''
        return [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    def human_face_detector_from_path(self, img_path: str):
        '''
        Uses haar cascade classifier to identify human faces
        
        input:
            img_path (str): path to image
            
        output:
            human_face (bool)
        '''
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('cnn/haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def human_face_detector(self, img_bytes: bytes):
        '''
        Uses haar cascade classifier to identify human faces
        
        input:
            img_bytes (bytes): image in bytes format
            
        output:
            human_face (bool)
        '''
        decoded = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('cnn/haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0
    
    def create_model(self):
        '''
        Creates, trains, tests and then saves the model.
        Uses resnet features available in resnet.npy if present, else generates them and save them.
        '''
        
        
        # load train, test, and validation datasets
        self.load_or_create_datasets()
            
        self.print_dataset_info(self.train_targets, 'train')
        self.print_dataset_info(self.valid_targets, 'valid')
        self.print_dataset_info(self.test_targets, 'test')

        self.init_custom_model()

        self.dog_model.summary()
        self.dog_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
        datagen_resnet_train.fit(self.train_resnet)
        datagen_resnet_valid.fit(self.valid_resnet)

        epochs = 200
        batch_size = 20
        self.dog_model.fit_generator(datagen_resnet_train.flow(self.train_resnet, self.train_targets, batch_size=batch_size),
                            steps_per_epoch=self.train_resnet.shape[0] // batch_size,
                            epochs=epochs, verbose=1, callbacks=[checkpointer],
                            validation_data=datagen_resnet_valid.flow(self.valid_resnet, self.valid_targets, batch_size=batch_size),
                            validation_steps=self.valid_resnet.shape[0] // batch_size)
        
        self.dog_model.load_weights(MODEL_PATH) 
        
        # report test accuracy
        resnet_predictions = [np.argmax(self.dog_model.predict(np.expand_dims(feature, axis=0))) for feature in self.test_resnet]
        test_accuracy = 100*np.sum(np.array(resnet_predictions)==np.argmax(self.test_targets, axis=1))/len(resnet_predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
        
    def load_or_create_datasets(self):
        '''
        Loads datasets from resnet.npy if available, else creates them from the images and then saves them to resnet.npy.
        '''
        try:
            with open('resnet.npy', 'rb') as f:
                self.train_resnet = np.load(f)
                self.train_targets = np.load(f)
                self.valid_resnet = np.load(f)
                self.valid_targets = np.load(f) 
                self.test_resnet = np.load(f)
                self.test_targets = np.load(f) 
                
        except:
            print('resnet.npy file not found, generating features')
            self.train_files, self.train_targets = self.load_dataset('cnn/dogImages/train')
            self.valid_files, self.valid_targets = self.load_dataset('cnn/dogImages/valid')
            self.test_files, self.test_targets = self.load_dataset('cnn/dogImages/test')
            
            self.train_tensors = self.paths_to_tensor(self.train_files)
            self.valid_tensors = self.paths_to_tensor(self.valid_files)
            self.test_tensors = self.paths_to_tensor(self.test_files)
            
            self.train_resnet = np.vstack([self.extract_Resnet50(tensor) for tensor in self.train_tensors])
            self.valid_resnet = np.vstack([self.extract_Resnet50(tensor) for tensor in self.valid_tensors])
            self.test_resnet = np.vstack([self.extract_Resnet50(tensor) for tensor in self.test_tensors])

            with open('resnet.npy', 'wb') as f:
                np.save(f, self.train_resnet)
                np.save(f, self.train_targets)
                np.save(f, self.valid_resnet)
                np.save(f, self.valid_targets)
                np.save(f, self.test_resnet)
                np.save(f, self.test_targets)
                
    def init_custom_model(self):
        self.dog_model = Sequential()
        self.dog_model.add(GlobalAveragePooling2D(input_shape=self.train_resnet.shape[1:]))
        self.dog_model.add(Dense(532, activation='relu'))
        self.dog_model.add(Dropout(0.3))
        self.dog_model.add(Dense(266, activation='relu'))
        self.dog_model.add(Dropout(0.1))
        self.dog_model.add(Dense(133, activation='softmax'))
        
        
    def load_dataset(self, path: str):
        '''
        Load image datasets from path
        
        input:
            path (str): path to directory containing the images
            
        output:
            dog_files (list[str]), dog_targets (list[str])
        '''
        data = load_files(path)
        dog_files = [filename for filename in np.array(data['filenames']) if filename.endswith('.jpg')]
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets

    def print_dataset_info(self, targets: list[str], save_name: str=None):
        '''
        Print or save dataset plot
        
        input:
            targets (list[str]): image targets
            save_name (str): optional file name
        '''
        dogs, counts = np.unique(np.argmax(targets, axis=1), return_counts=True)
            
        plt.figure(figsize=(30,60))
        sns.barplot(x='Count', y='Breed', data={'Breed': [self.get_dog_name(int(dog)) for dog in dogs], 'Count': counts}, palette="Blues_d")
        plt.title('Number of images per breed in training set')
        
        if save_name:
            plt.savefig('{}.png'.format(save_name))
