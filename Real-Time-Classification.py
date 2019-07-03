from time import sleep
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,jsonify, Response, session
from celery import Celery
from flask import current_app
import time, os
import torch
import pickle
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
from nocache import nocache

# Setting the scientific mode off
np.set_printoptions(suppress=True, precision= 2)
torch.set_printoptions(sci_mode=False)

# Choosing GPU vs CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convnet Class definition
class ConvNet(nn.Module):
  def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None, dropout_p=0.3):
      super(ConvNet, self).__init__()
      layers = []
      cur_size = 32
      cur_filter_size = 3
      last_channel_count = input_size
      for i, current_channel_count in enumerate(hidden_layers):
          cur_padding = cur_size - 1 + cur_filter_size
          cur_padding = int((cur_padding - cur_size) / 2 )
          layers.append(nn.Conv2d(in_channels=last_channel_count, out_channels=current_channel_count, kernel_size=(3, 3), padding=cur_padding))
          if norm_layer:
            layers.append(nn.BatchNorm2d(current_channel_count))
          layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
          layers.append(nn.ReLU())
          layers.append(nn.Dropout(p=dropout_p))
          last_channel_count = current_channel_count
          cur_size = int(cur_size / 2)
      layers.append(Flatten())
      layers.append(nn.Linear(in_features=hidden_layers[-1], out_features=num_classes))
      self.layers = nn.Sequential(*layers)
  def forward(self, x):
    forward_out = self.layers(x)
    return forward_out
      
# Class to flatten the input to it
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Variable Declarations
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 1024]
num_epochs = 30
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
norm_layer = True


def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                     ])
    
test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False
                                           ,
                                            
                                          transform = test_transforms
                                          )

# Test data 
target = np.asarray(unpickle('datasets/cifar-10-batches-py/test_batch')[b'labels'])
test_images = np.asarray(unpickle('datasets/cifar-10-batches-py/test_batch')[b'data'])
labels = unpickle('datasets/cifar-10-batches-py/batches.meta')[b'label_names']
print(target)

# Storing all indices of each class' samples
airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck = [],[],[],[],[],[],[],[],[],[]
for i in range(len(target)):
    if target[i] == 0:
        airplane.append(i)
    if target[i] == 1:
        auto.append(i)
    if target[i] == 2:
        bird.append(i)
    if target[i] == 3:
        cat.append(i)
    if target[i] == 4:
        deer.append(i)
    if target[i] == 5:
        dog.append(i)
    if target[i] == 6:
        frog.append(i)
    if target[i] == 7:
        horse.append(i)
    if target[i] == 8:
        ship.append(i)
    if target[i] == 9:
        truck.append(i)
    
#print(airplane)
        
# Defining a skeleton model
cifar_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer, dropout_p=0.1).to(device)

# Loading the trained model
loaded_model = torch.load("final_1024_0.1.ckpt", map_location='cpu')
print("Model loaded")

# Loading the static dictionary from the trained model into the skeleton model
cifar_model.load_state_dict(loaded_model["model"])
cifar_model
cifar_model.eval()
#plot_CIFAR(0)
with torch.no_grad():
    output = cifar_model(test_dataset[5548][0].unsqueeze_(0))
    _, predicted = torch.max(output.data, 1)
    print(predicted, output, test_dataset[5548][1])



app = Flask(__name__)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Initializing')


# Flask Configuration
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

# Redis-Flask-Cerley config initialization

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend = app.config['CELERY_RESULT_BACKEND'])


# Method executed when "Select Random Image" clicked on web interface. This calculates a random index from the selected class
# and shows the image sample on the web application.
@app.route('/getTarget',methods=['GET','POST'])
def getTarget():
    print("Getting target")
    if request.method == 'POST':
        target_image = request.form['selectedItem']
        print(target_image)
        
    if target_image == '0':
        
        index = random.choice(airplane)
        print("index0", index)
    if target_image == '1':
        
        index = random.choice(auto)
        print("index1", index)
    if target_image == '2':
        
        index = random.choice(bird)
        print("index2", index)
    if target_image == '3':
        
        index = random.choice(cat)
        print("index3", index)
    if target_image == '4':
        
        index = random.choice(deer)
        print("index4", index)
    if target_image == '5':
        
        index = random.choice(dog)
        print("index5", index)
    if target_image == '6':
        
        index = random.choice(frog)
        print("index6", index)
    if target_image == '7':
        
        index = random.choice(horse)
        print("index7", index)
    if target_image == '8':
        
        index = random.choice(ship)
        print("index8", index)
    if target_image == '9':
        
        index = random.choice(truck)
        print("index9", index)
    file1 = open("MyFile.txt","w") 
    file1.write(str(index))
    file1.close()
    arr = test_images[index]
    R = arr[0:1024].reshape(32,32)/255.0
    G = arr[1024:2048].reshape(32,32)/255.0
    B = arr[2048:].reshape(32,32)/255.0
 
    img = np.dstack((R,G,B))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(img,interpolation='bicubic')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "newplot" + timestr
    plt.imsave('static/images/' + filename + '.png', img)
    #plt.savefig(img, format= 'png')
    
    
    #image_url = base64.b64encode(img.getvalue())
    return render_template('index.html' , filename = 'images/' + filename + '.png')
   



# The background worker, which is invoked from the route "/calculating" (defined below. The Loaded model predicts the class and returns the 
# probabilities of each class
@celery.task(bind = True)
def classifyImage(self):
    with app.app_context():
        file1 = open("MyFile.txt","r") 
        target_index = file1.read()
        file1.close()
        print(target_index)
        self.update_state(state='PROGRESS', meta = {'result' : int(-1)}) 
        result = -1
        print('entered')
        with torch.no_grad():
            output = cifar_model(test_dataset[int(target_index)][0].unsqueeze_(0))
            print("target" , int(target_index))
            print("label", test_dataset[int(target_index)][1])
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().detach().numpy()
            print(predicted, output)
            prob = nn.Softmax(dim =1)
            probs  = prob(output)
            probs = probs.cpu().detach().numpy()
        self.update_state(state='COMPLETE', meta = {'result' : int(predicted[0]), 'airplane' : float(probs[0][0]), 'automobile' : float(probs[0][1]), 'bird' : float(probs[0][2]),
                                                     'cat' : float(probs[0][3]), 'deer' : float(probs[0][4]), 'dog' : float(probs[0][5]), 'frog' : float(probs[0][6]),
                                                     'horse' : float(probs[0][7]), 'ship' : float(probs[0][8]), 'truck' : float(probs[0][9])})
    return {'result' : int(predicted[0]), 'airplane' : float(probs[0][0]), 'automobile' : float(probs[0][1]), 'bird' : float(probs[0][2]),
                                                 'cat' : float(probs[0][3]), 'deer' : float(probs[0][4]), 'dog' : float(probs[0][5]), 'frog' : float(probs[0][6]),
                                                 'horse' : float(probs[0][7]), 'ship' : float(probs[0][8]), 'truck' : float(probs[0][9])}
    

# Invoked when "Classify" is clicked on the web interface
@app.route('/calculating', methods=['GET','POST'])
def calculating():
    print('calculating...')
    
    
    task = classifyImage.apply_async()
    
    return jsonify({}), 202, {'Result': url_for('result',
                                                  task_id=task.id)}

    
# Returns the results to the Frontend
@app.route('/result/<task_id>')
def result(task_id):
    task = classifyImage.AsyncResult(task_id)
    print(task.state)
    if task.state == 'PROGRESS':
        print('Job started',task.state)
        response = {'state'  : task.state}
    elif task.state != 'FAILURE' and (task.state == 'SUCCESS' or task.state == 'COMPLETE'):
        print('Job finsihed')
        response = {'state'  : task.state,
                    'result' : task.info.get('result', 0),
                    'airplane': task.info.get('airplane', 1),
                    'automobile': task.info.get('automobile', 2),
                    'bird': task.info.get('bird', 3),
                    'cat': task.info.get('cat', 4),
                    'deer': task.info.get('deer', 5),
                    'dog': task.info.get('dog', 6),
                    'frog': task.info.get('frog', 7),
                    'horse': task.info.get('horse', 8),
                    'ship': task.info.get('ship', 9),
                    'truck': task.info.get('truck', 10),
                    
                    'status': str(task.info)}
        print(response)
    else:
        # Something went wrong in the background job
        # This is the exception raised
        response = {
            'state': task.state,
            'status': str(task.info) 
        }
    return jsonify(response)    
    


# Default Route (Index.html)
@app.route("/")
def index():
    return render_template('index.html', filename = 'images/default.jpg')

if __name__ == "__main__":
    app.run()

    