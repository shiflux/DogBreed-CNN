from cnn.dog_breeder_model import DogBreeder

dog_breed = DogBreeder()

print('Predicting Australian Shepherd..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/Australian-Shepherd.jpg')))

print('Predicting Border Collie..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/border-collie.jpeg')))

print('Predicting Akita Inu..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/Taka_Shiba.jpg')))

print('Predicting Husky..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/husky.jpeg')))

print('Predicting actor Keanu Reeves..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/keanu.jpg')))

print('Predicting actor Jason Momoa..')
print(dog_breed.get_dog_name(dog_breed.predict_dog_from_path('cnn/test_images/momoa.jpg')))