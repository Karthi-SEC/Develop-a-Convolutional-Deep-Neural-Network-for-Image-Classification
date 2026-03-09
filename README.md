# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the dataset from the tensorflow library.

### STEP 2: 
Preprocess the dataset.



### STEP 3: 
Create and train your model.


### STEP 4: 
Include the training loss, validation loss vs iteration plot.



### STEP 5: 
Test the model for your handwritten scanned images.



### STEP 6: 
Create and train your model.



## PROGRAM

### Name: D Karthikeyan

### Register Number: 212224230115

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)

        return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
  print('Name:  D KARTHIKEYAN')
  print('Register Number:   212224230115')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # write your code here

      print('Name:  D KARTHIKEYAN')
      print('Register Number:   212224230115')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

<img width="665" height="420" alt="image" src="https://github.com/user-attachments/assets/490bffc0-9225-45f6-8eef-2d6576633044" />


Include the Training Loss per epoch

## Confusion Matrix

<img width="709" height="608" alt="image" src="https://github.com/user-attachments/assets/065bd6c0-508e-457e-9a10-4b15209ee51f" />


## Classification Report
<img width="812" height="562" alt="image" src="https://github.com/user-attachments/assets/72852fc6-9538-44a5-851b-7fdbfb29faf9" />


### New Sample Data Prediction
<img width="719" height="762" alt="image" src="https://github.com/user-attachments/assets/5fc833de-3c35-4699-a28e-4f8975766c1e" />


## RESULT

The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
