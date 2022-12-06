from cnn.dog_breeder_model import DogBreeder

dog_breed = DogBreeder()

print('Predicting Australian Shepherd..')
dog_breed.predict_dog('cnn/test_images/Australian-Shepherd.jpg')

print('Predicting Border Collie..')
dog_breed.predict_dog('cnn/test_images/border-collie.jpeg')

print('Predicting Akita Inu..')
dog_breed.predict_dog('cnn/test_images/Taka_Shiba.jpg')

print('Predicting Husky..')
dog_breed.predict_dog('cnn/test_images/husky.jpeg')

print('Predicting actor Keanu Reeves..')
dog_breed.predict_dog('cnn/test_images/keanu.jpg')

print('Predicting actor Jason Momoa..')
dog_breed.predict_dog('cnn/test_images/momoa.jpg')