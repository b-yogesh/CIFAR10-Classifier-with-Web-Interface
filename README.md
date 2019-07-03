# CIFAR10-Classifier-with-Web-Interface

To start the web interface
# Clone this repo
# Train the model from CIFAR10Final.ipynb
# The pickled model must be stored in the same directory as of Real-Time-Classification.py

# Install celery
# Install Redis

# Start the Redis server
# Start the celery worker 
  - celery worker -A Real-Time-Classification.celery
# Start the Flask python file
  - python Real-Time-Classification.py

# Webpage rendered on localhost:5000

