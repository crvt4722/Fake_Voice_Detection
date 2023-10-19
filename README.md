# By Pham Van Toi
## Description
This repository is used to classify the audio which is the real or fake voice from robots, AI computers, ... by the CNN model.   
## Idea
First, we have to convert the audio files (.wav) to the image files (.png). You can find it out in this [source](processing_data.py).  

From these data, we create a CNN model and save it to [model.keras](model.keras) file.  

From this trained model(model.keras), we can use it to classify the audio which is the real or fake voice.  

## Set up
All needed datasets, I shared in the [data](data/) folder.  

To apply this code, you have to install the needed libraries first by running this command:  
`pip install -r setup.txt`  

*Have an enjoyable coding!*
