"""
Description: Library for the functions of the Mexican Sign Language Translator
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""
# Import standar libraries
import cv2
import numpy as np
import tensorflow
import os
import csv
from tensorflow.keras.models import *
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from datetime import datetime

def mediapipe_detection(image, model):
    image.flags.writeable = False                  # Image is no longer writeable
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    #Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    
def draw_landmarks(image, results, mp_drawing, mp_holistic):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections

##Actualizado
def extract_keypoints(results):
    ## Detectar puntos cuerpo ( pose)
    count = 0
    pose = np.array([])
    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            pose = np.append(pose,np.array([[res.x, res.y]])).flatten()
            count += 1
            if count == 15:
                break
    else: pose = np.zeros(15*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose, lh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    cv2.putText(output_frame, actions[np.argmax(res)], (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(output_frame, str(round(res[np.argmax(res)]*100,2)), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    return output_frame

def column_names(mp_hands):
    # Crear Dataframe con los nombres de las columnas
    coords = ["x", "y"]

    ## Lista de nombres de columnas
    list_column_names = []
    ## PALABRA
    list_column_names.append("Sign")
    
    ## MANO IZQUIERDA
    count=0
    # Codigo para acceder a los puntos de referencia
    for landmark in mp_hands.HandLandmark:
        for coord in coords:
            list_column_names.append("L"+str(landmark) + "_" + str(count) + "_" + coord)
        count+=1   
        
    return list_column_names

def resize_img(img,cap,new_width):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = height/width
    new_height = int(ratio*new_width)
    resized_img = cv2.resize(img,(new_width,new_height))
    return resized_img, new_height

#Function to create CSV file or add dataset to the existed CSV file
def toCSV(filecsv,keypoints, mp_holistic):
    
    if os.path.isfile(filecsv):
        #print("File exist thus shall write append to the file")
        with open(filecsv, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(keypoints)
    else:
        #print("File not exist thus shall create new file as", filecsv)
        with open(filecsv, 'w', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            list_column_names = column_names(mp_holistic)
            writer.writerow(list_column_names)
            writer.writerow(keypoints)

def scale_points(x):
    x = x.astype("float")
    x_points = x[0::2]
    y_points = x[1::2]
    x_max=np.max(x_points)
    x_min=np.min(x_points)
    y_max=np.max(y_points)
    y_min=np.min(y_points)
    width = (x_max-x_min)
    height=(y_max-y_min)
    if height!=0 or width!=0:
        keypoints_x=np.divide(np.subtract(x_points,x_min),width)
        keypoints_y=np.divide(np.subtract(y_points,y_min),height)  
    
        return list(zip(keypoints_x, keypoints_y))
    else: return x

def model_creation(num_labels, X_train, y_train, X_test, y_test, epochs, train, **pretrained_model):
    if train==0:
        model = load_model(pretrained_model)
        model.summary()
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))
    
    else:
        model = Sequential()
        model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30,126)))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(GRU(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_labels, activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        filepath = '../saved_data/my_best_model_{}.h5'.format(datetime.now())
        checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
        model.fit(X_train, y_train, epochs=epochs, callbacks=checkpoint)
        
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))

        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test)

        # Print the test loss and accuracy
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
    return model

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tensorflow.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    
    return data_augmentation