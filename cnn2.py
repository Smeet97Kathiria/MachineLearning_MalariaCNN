import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from IPython.display import SVG
from keras.utils import model_to_dot

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

image_Array = np.load('Img_Array.npy')
label_Array = np.load('Lbl_Array.npy')

# 20% of data for testing dataset
X_train, X_test, y_train, y_test = train_test_split(image_Array, label_Array, train_size=0.80, random_state=97)

# 20% of data for validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(y_test), random_state=97)

dense_Layers = [3]
layer_Sizes = [32]
cnn_Layers = [4]
optimizer = Adam(lr=1e-3)

for dense_layer in dense_Layers:
	for layer_size in layer_Sizes:
		for conv_layer in cnn_Layers:
			
			model_Info = "{}-cnnLayer-{}-Nodes-{}-denseLayer-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			print(model_Info)

			cnn_model = Sequential()

			cnn_model.add(Conv2D(layer_size, (3, 3), input_shape=image_Array.shape[1:]))

# passing weiighted sum to an activation function to transfrom the output between 0 and 1
			cnn_model.add(Activation('relu'))
			cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
			
			for l in range(conv_layer - 1):
				if l > 1:
					cnn_model.add(Conv2D(64, (3, 3)))					
				cnn_model.add(Conv2D(layer_size, (3, 3)))
				cnn_model.add(Activation('relu'))
				cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

			cnn_model.add(Flatten())
			
			for _ in range(dense_layer):
				cnn_model.add(Dense(64))
				cnn_model.add(Dropout(0.3))
				cnn_model.add(Activation('relu'))	

			cnn_model.add(Dense(1))
			cnn_model.add(Dropout(0.3/2))
			cnn_model.add(Activation('sigmoid'))

			tensorboard = TensorBoard(log_dir="logs/{}".format(model_Info))
			cnn_model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		
	

			cnn_model.fit(X_train, y_train,
					batch_size=64,  
					epochs=18, 
					validation_data=(X_val, y_val),
					callbacks=[tensorboard])
	
cnn_model.save('Malaria-CNN.model')

print(cnn_model.summary())

for layer in cnn_model.layers:
    
    for index,layer in enumerate(cnn_model.layers):
        print('The shape of model output at layer {i} is:-'.format(i = index))
        print(layer.output_shape)
    break   
print('------------------------------------------------------------\n------------------------------------------------------------')
for layer in cnn_model.layers:
     for index,layer in enumerate(cnn_model.layers):
        print('The shape of model input at layer {i} is:-'.format(i = index))
        print(layer.input_shape)
     break 
 

plot_model(cnn_model, to_file='model.png')

#Model Evaluation
training = cnn_model.history
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('Model-Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plotting the histroy of model and validation loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('Model-Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

def FT_PositiveRate_Getter(y_True, y_Prob, boundary):

    y_Predicted = np.fromiter([1 if x > boundary else 0 for x in y_Prob ], int)
    n_Positives = y_True.sum()
    n_Negatives = y_True.shape[0] - n_Positives
    
    
    n_true_pos = 0  # get n true positives
    n_false_pos = 0  # get n false positives
    for predicted_V,true_V in zip(y_Predicted,y_True):
        
        if true_V == 1 and predicted_V == 1:    # true positive
            n_true_pos += 1
       
        elif true_V == 0 and predicted_V == 1:    # false positive
            n_false_pos += 1
    t_positive_rate = n_true_pos/n_Positives
    f_positive_rate = n_false_pos/n_Negatives
    return f_positive_rate,t_positive_rate


def build_ConfusionMatrix(y_True,y_Prob,boundary):
    confusion_Matrix = np.array([[0,0],[0,0]])
    for predicted_V,true_V in zip(y_Prob,y_True):
        if true_V == 1:
            #true positive
            if predicted_V > boundary:
                confusion_Matrix[0,0] += 1
            #false negative
            else:
                confusion_Matrix[1,0] += 1
        else:
            #false positive
            if predicted_V > boundary: 
                 confusion_Matrix[0,1] += 1
            #true negative
            else:
                confusion_Matrix[1,1] += 1       
    fig = plt.figure(figsize=(5,5))
    axes =  fig.gca()
    sns.heatmap(confusion_Matrix,ax=axes,cmap='Accent',annot=True,fmt='g',
               xticklabels = ['Infected','Uninfected'],
               yticklabels=['Infected','Uninfected'])
    axes.set_ylabel('Actual',fontsize=19,color = "Black")
    axes.set_xlabel('Predicted',fontsize=19,color = "Orange")
    plt.title('Confusion Matrix',fontsize=22, color = "Blue")
    plt.show()

#Creating Auc-Roc Plots for checking results accuracy
y_predict = cnn_model.predict(X_test)
boundaries = np.arange(0.01, 1.01, 0.01)
boundaries = np.append(np.array([0, 0.00001, 0.001]), boundaries)

roc_AUC = np.array([FT_PositiveRate_Getter(y_test, y_predict, n) for n in boundaries])
roc_AUC = np.sort(roc_AUC, axis=0)
roc_AUC_Value = roc_auc_score(y_test, y_predict)

loss, accuracy = cnn_model.evaluate(X_test, y_test)
accuracy = accuracy
loss = loss
text = 'AUC-ROC score = {:.3f}'.format(roc_AUC_Value)
text += '\nAccuracy = {:.3f}'.format(accuracy * 100)
text += '\nLoss = {:.3f}'.format(loss)

plot_Figure = plt.figure(figsize=(7,7))

axes = plot_Figure.gca()
axes.set_title('Malaria AUC-ROC Curve',fontsize=28)
axes.set_ylabel('True Positive Rate',fontsize=20,color = "Blue")
axes.set_xlabel('False Positive Rate',fontsize=20,color = "Red")
axes.plot(roc_AUC[:,0],roc_AUC[:,1])
axes.text(s=text,x=0.25, y=0.5,fontsize=20, color = "Green")
plt.show()

# Creating confusion matrix with a threshold of 0.5
build_ConfusionMatrix(y_test,y_predict,0.5)



