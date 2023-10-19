from keras.models import load_model
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.metrics import accuracy_score, f1_score

# Function to convert waw to png
def append_ext(fn):
    fn = fn.replace(".wav",".png")
    fn = fn.split('/')[-1]
    return fn

# Load the testing data
testdf=pd.read_csv('data/test_data.csv',dtype=str)
testdf["voice_location"]=testdf["voice_location"].apply(append_ext)

test_data_path='data/test/'



# Initial the testing data
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=test_data_path,
    x_col="voice_location",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))

# TInh so buoc test
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# Tien hanh predict
test_generator.reset()

# Load the trained model
model = load_model('model.keras')


pred = model.predict(test_generator)

# Lay class predict probality lon nhat
predicted_class_indices=np.argmax(pred,axis=1)

# Load labels from the pickle file.
with open('model_indices.pickle', 'rb') as handle:
    labels = pickle.load(handle)

# Print the Accuracy and F1-Score metrics to the console.
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print('Accuracy: '+ str(accuracy_score(testdf['label'], predictions)))
print('F1-score: '+ str(f1_score(testdf['label'], predictions,average='binary', pos_label='1')))







