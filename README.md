## Introduction

**COMPYSYS 302 Project 2 : Sign-Sys by Group 21**

Sign-sys is a sign language recognition software that is created to help deaf users interact with automated home and 
security systems using sign language gestures.

## Team Members:

- Sai Kiran N Kamat (skam727)
- Harsh Thorat (htho884)
- Lojanan Sivanantharuban (lvis157)

## Getting Started

To get started you must have installed:
  - Python Version 3.12
  - A Virtual Environment like PyCharm to ensure smooth execution without any conflicts

## Running Sign-Sys
_(Before running the application, users must ensure to install the necassry packages used by the application files)_

Open the project in PyCharm and run the file **main.py**. This will open the homepage for Sign-Sys.

On clicking on the **Enter** button, the user will be taken to a page with 4 options:
- Load Data
- View Data
- Train
- Test

### Load data
Users must load a **.csv** file that should contain the dataset that they want to train. Note that the users cannot select any of the other 3 options unless they load a valid dataset first.

### View Data
Once the data is successfully loaded, the users can now view the loaded data in grayscale image format in a scrollable grid layout that displays 10 images in each row. There is also a search bar at the top of the window, which the users can use to search any label from **0 to 35**, upon which the images with the searched label are filtered and returned.

### Train
Once the data is successfully loaded, the users can click on **Train**, which takes them to a new window.
Here, the users can then choose to train any 1 of the 3 Models:
- AlexNet
- InceptionV3
- Sign-Sys Model

Along with the ability to select desired parameters:
- Train-Test Ratio
- Batch Size
- Epochs

Once the users have chosen their desired model and parameters, they can click on the **Start Training** Button to being the training process.

Once the model is trained, the results are stored into a **.pth** file, with the name of the model and the accuracy rate as the file title in a folder called Models within the project working directory.

### Test
_Please Note: In order to test the dataset, users must choose a trained model from Sign-Sys Project Working Directory_
Once the model is finished training, the users can click on the **Test** button to choose a trained model.
After choosing the trained model, the users will be taken to a new window, where they can view all the images from their dataset in a window with multiple pages to navigate through by clicking Previous or Next.
When the users click on any of the images, the software will open a small window with the image, the label prediction for that image and the probability graph for the predicted label.

#### Camera
Within the test window, users have the option to check the predictions in real time using their webcam by clicking on the **Camera** button.
This will open a window with the user's webcam capture and the user can then press **c to capture** a particular hand-sign or **q to quit** on their keyboard.
After capturing a photo, the software will then predict the label for the captured image of the hand-sign.

## Screenshots
1. Home Page

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/4cf61ca5-2fc3-4517-b2d1-d21df97ff1ee)

2. User Options

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/425eb38e-f0bd-4b6f-bd91-083971710a8e)

3. Loaded Data

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/9537285a-3665-4c31-8db5-cd0cbb1edabe)

4. View Data

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/d609d870-86fa-4569-bae3-d2bb5148026c)

5. Train Options

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/c377ec77-475f-4417-87dc-52f871b9054c)

6. Post Training

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/46729191-e911-4b40-82f0-62c79799a172)

7. Test

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/349a1823-8793-491e-b44d-3f29c2d30425)

8. Prediction Probabilities

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/bd08cb92-3153-47be-ba6a-e0947634bbae)

9. Camera Input

![image](https://github.com/COMPSYS302/project-python-group-21/assets/126769938/9468e2c3-a52c-4e5e-a0a1-fc9eafc32321)
