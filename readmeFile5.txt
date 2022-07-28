Introduction:

In this project, I have deployed a model that detects whether a person is having 
Pneumonia or not. The image classification is done by using Convolution Neural 
Network (CNN). CNN is an artificial neural network that has the ability to detect 
patterns in the images.
Medical image analysis is a deep learning project in which medical images are analysed 
using CNN. CNN helps us to classify the images on the basis of certain patterns. In this 
model I have chosen a dataset having Chest X-Ray images.

Motivation:

Pneumonia is an acute respiratory infection that affects the lungs. It is a fatal illness in 
which the air sacs get filled with pus and other liquid . There are mainly two types of 
pneumonia: bacterial and viral. Generally, it is observed that bacterial pneumonia 
causes more acute symptoms. The most significant difference between bacterial and 
viral pneumonia is the treatment. Treatment of bacterial pneumonia is done using 
antibiotic therapy, while viral pneumonia will usually get better on its own . It is a 
prevalent disease all across the globe. Its principal cause includes a high level of 
pollution. Pneumonia is ranked 8 in the list of the top 10 causes of death in the United 
States . Due to pneumonia, every year, 3.7 lakh children die in India, which constitutes 
a total of fifty percent of the pneumonia deaths that occur in India . Children can be 
protected from pneumonia. It can be prevented with simple interventions and treated 
with low-cost, low-tech medication and care” . 
Therefore, there is an urgent need to do research and development on computer-aided 
diagnosis so that the pneumonia-related mortality, especially in children, can be 
reduced.
One of the following tests can be done for pneumonia diagnosis: chest X-rays, CT of 
the lungs, ultrasound of the chest, needle biopsy of the lung, and MRI of the chest . 
Currently, chest X-rays are one of the best methods for the detection of pneumonia

Objective:

To capture X-Ray images to analyse the presence of Pneumonia using deep learning 
techniques.
To classify X-Ray images as a Pneumonia cases or Normal cases with accuracy.
To detect the presence of Pneumonia.

Problem Statement:

In the medical field , Pneumonia is detected by Doctors by referring the X-Ray 
images which is very time consuming. Therefore , to overcome this problem , an 
alternative way is to design the system that will automatically identify the presence 
of Pneumonia in X-Ray images using Convolutional Neural Network and also 
provide faster and accurate solutions.

Software requirements:

Language used: Python 3.7 or above
Operating system: Windows 10 
Tool used: Google Collaboratory Notebook 

Hardware Requirements:

⚫ Processor: Intel core i5 or above.
⚫ 64-bit, quad-core, 2.5 GHz minimum per core 
⚫ Ram: 4 GB or more 
⚫ Hard disk: 10 GB of available space or more.
⚫ Display: Dual XGA (1024 x 768) or higher resolution monitors
⚫ Operating system: Windows

Technology Used:

Python- It is an interpreted, high-level, general-purpose programming language. 
Created by Guido van Rossum and first released in 1991, Python's design philosophy 
emphasizes code readability with its notable use of significant whitespace. Its language 
constructs and object-oriented approach aim to help programmers write clear, logical 
code for small and large-scale projects. Python is dynamically typed and 
garbage-collected. It supports multiple programming paradigms, including procedural, 
object-oriented, and functional programming. Python is often described as a "batteries 
included" language due to its comprehensive standard library.

Deep learning!

Deep learning is a subset of machine learning, which is essentially a neural network 
with three or more layers. These neural networks attempt to simulate the behavior of the 
human brain—albeit far from matching its ability—allowing it to “learn” from large 
amounts of data. While a neural network with a single layer can still make approximate 
predictions, additional hidden layers can help to optimize and refine for accuracy.
Deep learning drives many artificial intelligence (AI) applications and services that 
improve automation, performing analytical and physical tasks without human 
intervention. Deep learning technology lies behind everyday products and services 
(such as digital assistants, voice-enabled TV remotes, and credit card fraud detection) 
as well as emerging technologies (such as self-driving cars).

Algorithm Used

Convolutional neural networks(ConvNet/CNN)

A Convolutional Neural Network is a Deep Learning algorithm which can take in an 
input image, assign importance (learnable weights and biases) to various 
aspects/objects in the image and be able to differentiate one from the other. The 
pre-processing required in a ConvNet is much lower as compared to other classification 
algorithms. While in primitive methods filters are hand-engineered, with enough 
training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of 
Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. 
Individual neurons respond to stimuli only in a restricted region of the visual field 
known as the Receptive Field. A collection of such fields overlap to cover the entire 
visual area.
Convolutional neural networks are distinguished from other neural networks by their 
superior performance with image, speech, or audio signal inputs. They have three main 
types of layers, which are:

Convolutional layer-

⚫ Convolutional Layer
⚫ Pooling layer
⚫ Fully-connected (FC) layer

The convolutional layer is the core building block of a CNN, and it is where the 
majority of computation occurs. It requires a few components, which are input data, a 
filter, and a feature map. Let’s assume that the input will be a color image, which is 
made up of a matrix of pixels in 3D. This means that the input will have three 
dimensions—a height, width, and depth—which correspond to RGB in an image. We 
also have a feature detector, also known as a kernel or a filter, which will move across 
the receptive fields of the image, checking if the feature is present. This process is 
known as a convolution.
The feature detector is a two-dimensional (2-D) array of weights, which represents part 
of the image. While they can vary in size, the filter size is typically a 3x3 matrix; this 
also determines the size of the receptive field. The filter is then applied to an area of the 
image, and a dot product is calculated between the input pixels and the filter. This dot 
product is then fed into an output array. Afterwards, the filter shifts by a stride, 
repeating the process until the kernel has swept across the entire image. The final output 
from the series of dot products from the input and the filter is known as a feature map, 
activation map, or a convolved feature.

Pooling Layer

Pooling layers, also known as downsampling, conducts dimensionality reduction, 
reducing the number of parameters in the input. Similar to the convolutional layer, the 
pooling operation sweeps a filter across the entire input, but the difference is that this 
filter does not have any weights. Instead, the kernel applies an aggregation function to 
the values within the receptive field, populating the output array.
 
There are two main types of pooling:

Max pooling: As the filter moves across the input, it selects the pixel with the 
maximum value to send to the output array. As an aside, this approach tends to be used 
more often compared to average pooling.

Average pooling: As the filter moves across the input, it calculates the average value 
within the receptive field to send to the output array.
While a lot of information is lost in the pooling layer, it also has a number of benefits to 
the CNN. They help to reduce complexity, improve efficiency, and limit risk of 
overfitting. 

Fully-Connected Layer

The name of the full-connected layer aptly describes itself. As mentioned earlier, the 
pixel values of the input image are not directly connected to the output layer in partially 
connected layers. However, in the fully-connected layer, each node in the output layer 
connects directly to a node in the previous layer.
This layer performs the task of classification based on the features extracted through the 
previous layers and their different filters. While convolutional and pooling layers tend 
to use ReLu functions, FC layers usually leverage a softmax activation function to 
classify inputs appropriately, producing a probability from 0 to 1.

Types of convolutional neural networks
⚫ AlexNet 
⚫ VGGNet 
⚫ GoogLeNet
⚫ ResNet 
⚫ ZFNet

However, LeNet-5 is known as the classic CNN architecture.
Some common applications of CNN:

Marketing: Social media platforms provide suggestions on who might be in 
photograph that has been posted on a profile, making it easier to tag friends in photo 
albums. 
Healthcare: Computer vision has been incorporated into radiology technology, 
enabling doctors to better identify cancerous tumors in healthy anatomy.
Retail: Visual search has been incorporated into some e-commerce platforms, allowing 
brands to recommend items that would complement an existing wardrobe. 
 
Automotive: While the age of driverless cars hasn’t quite emerged, the underlying 
technology has started to make its way into automobiles, improving driver and 
passenger safety through features like lane line detection.
Process involved:
Step1: Loading the Dataset.
Step 2: Initializing the data Now, Here I imported some important libraries and define 
directory path.
Step 3: Preparing the data: 
a) Data augmentation
Image augmentation technique is used for increasing the size of image training dataset. 
This is done by flipping, horizontal or vertical shifting, zooming or adding some noises 
to the same images.
b) Loading the images: There is a class known as flow from directory offered by 
Image Data Generator which reads the images from folders.
Step 4: Applying CNN: 
The CNN architecture has convolutional layers which receives inputs and transform the 
data from the image and pass it as input to the next layer. This transformation is known 
as the operation of convolutional. We need TensorFlow and necessary libraries for 
CNN.
Step 5: Fitting the model 
a) Defining callback list I have used Early Stopping which is called to stop the epochs 
based on some metric and conditions. It helps to avoid overfitting the model. Reduce 
learning rate when a metric has stopped improving. This callback monitors a quantity 
and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is 
reduced.
b) Assigning class weights
Step 6: Training the model:
The EarlyStopping has stopped at 10th epoch at val_loss =38.8% and val_accuracy = 
75.8%.
Step 7: Evaluating the model:

Confusion matrix:
The upper left (TP) denotes the number of images correctly predicted as normal cases 
and the bottom right (TN) denotes the correctly predicted number of images as cases of 
pneumonia. As Pneumonia case, the upper right denotes the number of incorrectly 
predicted images but were actually normal cases and the lower left denotes the number 
of incorrectly predicted Normal case images but were actually Pneumonia case.
Classification Report
The four metrics are given as follows:
Precision = True Positives / (True Positives + False Positives) 
accuracy =
TP+TN
TP+TN+FP+FN
precision =
TP
TP+FP
recall =
TP
FP+FN
Recall = True Positives / (True Positives + False Negatives) 
F1 = (2 * Precision * Recall) / (Precision + Recall) 
At last, visualized the predicted images using percentages.

CONCLUSION AND FUTURE SCOPE 

We proposed a computerized method for the segmentation and identification of a 
Pneumonia using the Convolution Neural Network. The input images are read from the 
local device using the file path and converted into grayscale images. These images are 
pre-processed for the elimination of noises that are present inside the original image. 
The proposed model had obtained an good accuracy and yields promising results 
without any errors and much less computational time.
In the future, we will continue the research to explore more accurate classification
architectures to diagnose two types of pneumonia, viruses, and bacteria.We can even 
deploy the model using flask, django and many more to make it a working model for 
industry. According to the description discussed above, the CNN-based model is a 
promising method to diagnose the disease through X-rays.

References:

⚫ For dataset: https://kaggle.com
⚫ For resolving errors: https://stackoverflow.com/
⚫ For Understanding the topics :
⚫ www.google.com
⚫ www.youtube.com
⚫ www.medium.com
⚫ https://docs.python.or