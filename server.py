from fastapi import FastAPI, File
from cnn.dog_breeder_model import DogBreeder
from cnn.consts import DOG_NAMES
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

dogbreed = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

dogbreed.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dog_breed_model = DogBreeder()


@dogbreed.get("/api/dogs")
async def dogs():
    return DOG_NAMES

@dogbreed.post("/api/predict_dog")
async def predict_dog(file: bytes=File()):
    try:
        is_dog = dog_breed_model.dog_detector(file)
        
        if is_dog:
            is_human = False
        else:
            is_human = dog_breed_model.human_face_detector(file)
        
        if not (is_dog or is_human):
            print('Neither human nor dog detected!')
        
        breed_num = int(dog_breed_model.predict_dog(file))
        breed_name = dog_breed_model.get_dog_name(breed_num)

        return {
            'breed': {'name': breed_name, 'num': breed_num} if (is_dog or is_human) else None,
            'image_type': 'dog' if is_dog else 'human' if is_human else 'unknown'
            }
    except:
        return {
            'error': 'invalid_image'
        }
        
if __name__ == "__main__":
    uvicorn.run("server:dogbreed", host="127.0.0.1", port=8000, reload=True)