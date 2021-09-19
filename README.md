# Guidedog
GuideDog is an AI/ML-based mobile app designed to assist the lives of the visually impaired, 100% voice-controlled. You may as well think of it as "speaking guide dog," as the name suggests. It has three key features based on the scene captured by your mobile phone:
  1) Reads text upon command
  2) Describes the scene around you upon command
  3) Warns you if there is an obstacle in front of you

<h4>Android App</h4> 
  
  * UI/UX 
    * Simple and Responsive
    * Voice Assistant architecture for targeted audience
  
  * Libraries / APIs
    * GC Speech-to-text and Text-to-Speech
    * Android SDK , androidX
    * ML Kit object detection and tracking api
    * TensorFlow Lite MobileNet Image Classification Model   

![image](https://user-images.githubusercontent.com/52928325/133925137-7d486496-7ab8-425a-8f86-94324ecc6ccd.png)

<h4>Backend</h4>
  
  * Flask API
    * Image Captioning
    * Optical Character Recognition   
   
  * Deployment
    * Google App Engine
    * fast central API with different endpoints   

![image](https://user-images.githubusercontent.com/52928325/133925105-7bd1132e-0d78-4df0-9e4d-fda0db31b97b.png)
 


<h4> Image Captioning </h4>

We used tensorflow to build and train model for image captioning on MS-COCO 2014 based on the paper Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. The model uses standard convolutional network as an encoder to extract features from images (we use Inception V3) and feed the generated features into an attention-based decoder generate sentences. While the paper used LSTM model as a decoder, we use a simpler RNN instead. 

![image](https://user-images.githubusercontent.com/52928325/133925171-4130a140-6e93-4c5d-924d-7d49c9693214.png)

<h4>Get more insights : <a href="https://devpost.com/software/guidedog-smart-sunglasses-for-the-blind?ref_content=my-projects-tab&ref_feature=my_projects">Devpost</a> </h4>
