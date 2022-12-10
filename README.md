# [Dog Breed Finder](https://dogbreedfinder.app/)

![Screenshot](screenshot.png "Screenshot")
# Table of Contents
1. [Installation](README.md#installation)
2. [Run instructions](README.md#run-instructions)
3. [Project organization](README.md#file-description)
4. [Project motivation](README.md#project-motivation)
5. [Licensing, Authors and Acknowledgements](LICENSE)

## Installation
### 1. Clone repository

    `
    git clone git@github.com:shiflux/DogBreed-CNN.git
    `

### 2. Install python requirements

    '
    cd DogBreed-CNN
    pip3 install -r requirements.txt
    pip3 install tensorflow-cpu
    '

### 3. Install node dependencies

    '
    cd web/
    npm install
    '

## Run instructions
### 1.1 Run api service

    `
    python3 server.py
    `

### 1.2 Run web service

    `
    npm run dev
    `

### 2. Create and train model

    `
    python3 create_model.py
    `


## Project organization
- **cnn/** Contains the CNN model
- **web/** Contains web service built with NextJS

# Project motivation
This is a project for the Udacity "Data Scientist" nanodegree.

CNN models were the the part that struck me the most, and owning a dog, the choice of the project was a no brainer.