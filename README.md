# CIFAR10-Classifier-with-Web-Interface

Deep Convolutional Neural Network <br/>
Number of Layers - 5 <br/>
Epoch - 30 <br/>
Test Accuracy - 84.62 <br/>
<br/> 
To start the web interface <br/>
Clone this repo <br/>
Train the model from CIFAR10Final.ipynb <br/>
The pickled model must be stored in the same directory as of Real-Time-Classification.py <br/>
<br/> 
Install celery <br/>
Install Redis <br/>
 <br/>
 Start the Redis server <br/>
 Start the celery worker  <br/>
```
   celery worker -A Real-Time-Classification.celery
```
  <br/>
 Start the Flask python file <br/>
 
```
   python Real-Time-Classification.py
```
<br/>

 Webpage rendered on localhost:5000 <br/>
 <br/>
 **Screenshots included

