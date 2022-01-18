<p align="center">
<a href="https://dscommunity.in">
	<img src="https://github.com/Data-Science-Community-SRM/template/blob/master/Header.png?raw=true" width=80%/>
</a>
	<h2 align="center"> Face Mask Detection </h2>
	<h4 align="center"> This project aims to identify and perform binary classification to sort if a person is wearing a face mask or not. We have built a model that uses MobileNetV2 to do the same, which is a popular library when it comes to identifying similar features. <h4>
</p>

---
[![DOCS](https://img.shields.io/badge/Documentation-see%20docs-green?style=flat-square&logo=appveyor)](https://github.com/Data-Science-Community-SRM/Face-Mask-Detection/blob/master/README.md) 
  [![UI ](https://img.shields.io/badge/User%20Interface-Link%20to%20UI-orange?style=flat-square&logo=appveyor)](INSERT_UI_LINK_HERE)
	
## MATERIALS AND METHODS

### Description of the problem

Today, using a face mask is a mandatory preventive measure, and keeping the mouth, nose, and cheeks covered has now made people only recognizable by their eyes, eyebrows, and hair. This is a problem to the human eye, which tends to find similarities in several faces with similar features. This problem also affects artificial intelligence systems, as facial recognition systems are now widespread. They are used to unlock smartphones, laptops, access sensitive applications, and enter various workplaces. Current systems usually process information from the person's entire face, which is why technology must adapt to these new conditions. The same has been done to maintain the user's biosecurity and at the same time allow them to continue with the activities as naturally as possible.
The literature has shown that systems seek to identify whether people use it properly. These works have had excellent results. However, facial recognition using biosecurity material has not yet been explored. All of this motivated the present investigation, in which a detection system with an approach. A facial recognition algorithm in controlled environments, allowing personnel to be identified automatically without removing the face mask. This can be implemented as an access system to an institution or a home, but at a low cost. This is ensured by using open-source programming software and simple features that reduce computational expense. For this reason, the possibility of improving the adaptability of current facial recognition systems in the face of new circumstances has been established as a starting hypothesis.


### Data Collection

The data in this project has been collected from various sources including images found on Kaggle.

### System Development

The programming language in use is python, and the framework in use is Keras. The proposal is to design a system capable of identifying a person’s face, even if they are wearing a mask or not. The images are divided into training and testing sets with a test size of 0.20. The training data is obtained from images, while the model runs through simultaneous video input from the webcam of the user. A classification model based on the MobileNetV2 architecture and the OpenCv’s face detector is used with the aim of having a better precision and robustness, as it uses smaller models with a low latency and low parameterization power. This improves the performance of mobile models in multiple tasks and benchmarks, resulting in a better accuracy. It also retains the simplicity and does not require any special operator to classify multiple images and various detection tasks for mobile applications. Once the face of the person has been identified, in the third stage, facial recognition is carried out, for which a set of own observations is used that is built based on the faces of various people.

### Training of Facial Recognition Models

The training of the facial recognition model includes loading the face mask data set, then training the face mask classifier using Keras/Tensorflow and then serialize the face mask classifier to the disk. The data is split into testing and training sets using train_test_split from scikitlearn. This is the first stage of the project.

### Application of the Facial Recognition System

The application of the facial recognition system is the second stage of the project. It involves Loading the face mask classifier from the disk and and detecting faces from the image/video stream. From this stream, each face ROI is extracted  and the face mask classifies is apllied to each face mask ROI giving a positive (“Mask”) or negative (“No Mask”) result. This result is then shown on the screen to the user.
	
## Preview
![Training Loss and Accuracy Plot](https://github.com/yatin2901/facemask/blob/main/plot.png)

Figure above shows the training accuracy vs training loss graph.

![Project Implementation](https://github.com/yatin2901/facemask/blob/main/Model.png)

Image above shows the implementation of the model in real-life.
## Functionalities
1. This prototype system allows for the facial recognition of people with and without a mask and could be used as a low computational consumption proposal for personnel access control. 
2. It can be implemented on to a webpage or an application for publishing and public use.
3. It could also be worked upon to perform various other functions that require image processing and facial recognition, for example, it is possible to know their identity. This is if the face is within the selected database. 

<br>


## Instructions to run

* Pre-requisites:
	-  pip install tensorflow
	-  pip install sklearn
	-  pip install imutils
	-  pip install matplotlib
	-  pip install numpy
	-  pip install os
	-  pip install cv2
	-  import haarcascade_frontalface_default.xml for CascadeClassifier

* Training the model 
```bash
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\DELL\Desktop\facemask\dataset"
CATEGORIES = ["mask", "no_mask"]

print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
```

* Model Execution

```bash
from keras.models import load_model
import cv2
import numpy as np

model = load_model("C:/Users/DELL/Desktop/facemask/dataset/mask_detector.model")

face_clsfr=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(672,672))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(3,224,224,3))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
```

## Contributors

<table>
<tr align="center">


<td>

Yashowardhan Samdhani

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Face-Mask-Detection/blob/master/Yashowardhan%20Samdhani.jpg"  height="120" alt="Yashowardhan Samdhani">
</p>
<p align="center">
<a href = "https://github.com/YashowardhanSamdhani"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/yashowardhansamdhani/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td>

Daketi Yatin

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Face-Mask-Detection/blob/master/Daketi%20Yatin.jpg"  height="120" alt="Daketi Yatin">
</p>
<p align="center">
<a href = "https://github.com/yatin2901"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/daketi-yatin-a50683204">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>



<td>

Harikrishnaa

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/Face-Mask-Detection/blob/master/hari_photo.jpg"  height="120" alt="Harikrishnaa">
</p>
<p align="center">
<a href = "https://github.com/Harikrishnaa3131"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/hk3112/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
</tr>
  </table>
  
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with :heart: by <a href="https://dscommunity.in">DS Community SRM</a>
</p>

