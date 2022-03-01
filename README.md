# AWS-Deep-Learning-Challenge-2022
Training a deep learning model on the new Amazon EC2 DL1 instances powered by Habana Gaudi accelerators.

The model can be found at https://upload-aws-hack.s3.amazonaws.com/final_model.h5

## Inspiration

Whether we are scrolling through Instagram, Facebook, TikTok, or any other social media platform, we often come across celebrities, and more recently ‘influencers’, and at times envy their picture-perfect looks. It is within human nature to compare ourselves to our peers. However, social media seems to advantage of this human trait.

Most of the algorithms governing these sites are designed in such a fashion that only award the top few percentiles of the profiles on the platform by constantly recommending them to everyone else, as a result of which only a few selective profiles accumulate hundreds of thousands or even millions of ‘likes’, ‘followers’ and ‘subscribers’, i.e., the ‘awards’, while the median profile doesn’t show up much anywhere, if at all!

Such a winner-takes-all game creates a toxic environment for the average user that leads to people excessively relying on using filters and photoshop to manipulate and ‘improve’ the images they post. This ‘improvement’ could be as simple as blemishing some acne to completely re-engineering their faces using facial landmarks and GANs. Unfortunately, due to the sheer nature of the competition for attention on such sites, young people are sometimes tempted to undergo plastic surgery, just to improve the reach of their posts.

The biggest victims of this scheme of operations are adolescents and teenagers. Being exposed to this competition for attention at an early age can stoke anxiety around their body image and can deflate the person’s self-confidence. It can cause self-doubt, causing people to question: Am I good enough? People hold unrealistic standards and feel worse about themselves. It has been observed to take a toll on the teens’ mental health leading to disorders such as body dysmorphia as well as mood and eating disorders. 

Photoshop and filters that alter or edit images can also contribute to negative body image, says Jill M. Emanuele, Ph.D., the senior director of the Mood Disorders Centre at the Child Mind Institute. "Photoshop and filters present people and things in their best light," Emanuele says. "It creates a distorted fantasy world and raises the bar on what people perceive is 'the best way to be." [1](https://www.insider.com/how-social-media-affects-body-image)

Of all the photos you see on your social media feeds, there's a good chance most of them have been edited. According to a 2017 Harris Poll, nearly two-thirds of Americans edit their photos before posting. [2](https://www.globenewswire.com/news-release/2017/05/18/1312618/0/en/The-Filter-Effect-People-Distrust-Websites-Because-of-Manipulated-Photos.html) The isolation during the pandemic has made things worse, as many people have had limited in-person contact with friends and relatives and rely on social media much more than before. 

Therefore, it was of utmost importance to build a service that helps expose the fictitious nature of the pictures and photos posted on social media, and by extension, the internet; a tool that helps distinguish what’s real and what’s fake! Introducing FakeWeb.

## What it does

FakeWeb takes facial images sourced from social media sites, the web in general, or from the user's device and checks if it is a manipulated image or not. This manipulation may include but is not limited to using filters, manual or automated photoshopping, deepfakes, or even creating GAN-generated fake faces. If FakeWeb finds a facial image to be manipulated in one of the aforementioned ways, it detects the facial landmarks of the face in question and draws a face mesh around that face using those landmarks. 

If a photo is uploaded with multiple visible faces, FakeWeb scans through each one of these faces one by one and identifies if any or multiple of those faces have been morphed, and draws a face mesh around them accordingly. It then returns and displays the photo on the user-client so the user can visually perceive the extent of manipulation done in the image.

## How we built it

Firstly, we take the image fed to the application as an input and process it through a haar-cascade to identify and locate the faces within that image. We then crop those faces and pass them to the model to predict if they are original or modified. If the model predicts an image is manipulated, Google's Mediapipe project is triggered and is used to draw a mesh over the subject's face. Once all the images are processed in a similar way, the cropped (modified) faces are then pasted back on the base image. This image is then sent back to the client as a response and displayed. 

## Challenges we ran into

It was during this hackathon that we first tried to train a deep learning model on an AWS EC2 instance, let alone a DL1 type instance. We were unable to connect to the instances for almost a week after the commencement of the challenge and the approval of our credits and ended up wasting a handful of credits in the process. This was mainly due to our lack of understanding of how the AWS Cloud platform works. Training the model using the habana_frameworks.tensorflow module was another big mess and took us quite a while to figure out. But one of the biggest issues we faced was downloading the .h5 file of the model into our local machine. The error we faced quite often was that as soon as we would switch to a different terminal to download the files via the scp command, the session on the DL1 instance would expire or collapse and we would end up losing all the files on the session. In all, we had trained the same model seven times over until we figured out a way to transfer the files to an S3 bucket via the Amazon CLI. By then, we had burned up almost $187 worth of  AWS credits and it would literally have been our final attempt hadn't we successfully transferred our model then. Some of the other alternatives we had tried were using MobaXterm and Tmux to transfer the file but nothing seemed to work.

## What we learned and the accomplishments that we're proud of

We learned to train a Deep Learning model via transfer learning on a VGG16 model and then customizing Habana-Gaudi's ResNet-Keras model. One of our major accomplishments was to be able to deploy the entire backend on the cloud, especially when dealing with images since they are quite difficult to render on the client-side when a response is produced from a cloud-hosted backend. It was also for the first time that we were able to integrate a deep learning model with a fast API. We also, (eventually) learned to transfer the model from an EC2 instance to an S3 bucket. 

## What's next for FakeWeb

The future scope of FakeWeb holds a lot of possibilities. 

Firstly, we wish to increase the size of our training dataset by including the [FaceForensics++ dataset](https://github.com/ondyari/FaceForensics) while also inculcating elements from other open-source dataset collections as given [here](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121173) and [here.](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces) We had explored and studied these datasets extensively and even tried using them at some point of time during this challenge, although the time crunch and our questionable capacity to experiment with and test such a large dataset within our local systems led to us then rejecting the idea. This data will help us minimize or even eliminate the present biases of the model.

Next, as was originally planned for this project, we wish to create a web-extension version of this application that would allow a user to simply toggle a switch to parse all the images accessible on the present tab and feed them to the application directly. This can be easily achieved via web-scraping with [Beautiful-Soup](https://beautiful-soup-4.readthedocs.io/en/latest/) or [Selenium.](https://www.selenium.dev/) Once the images are processed and the response is prepared, the user will have to reload the page to check for any manipulations. The deep learning model to make the predictions will be replaced with a lighter ResNet—Keras one to handle the intensity of such a workload, the code for which we have already prepared and submitted to this challenge alongside our main model.

We also plan to add a feature that changes the color and the opacity of the face mesh created in accordance with the percentage probability that the classifier model returns as a prediction output. The higher the percentage probability of a face being a deepfake or being photoshopped, the opaquer a face mesh would be drawn. This could essentially be a slider from 0% opacity to 100% opacity depending on the value of the predictions drawn by the model. Also, the color of the face mesh could be changed in a similar fashion, where the mesh color shifts from one end of the spectrum to the other depending on the response produced by the model. 

With sufficient time at our hands in the approaching weeks, we are keen on continuing this project further and improving upon the model's accuracy as well as cleaning and segmenting the code. We also wish to work on the internal operations of the systems and handle the workflow more efficiently and effectively, so as to reduce the latency between the request and the response. 

## Training Images 
VGG16 Model -
![1](https://user-images.githubusercontent.com/65075827/156249085-e520dfe4-da49-45a8-9bde-0ef630880251.jpeg)

![2](https://user-images.githubusercontent.com/65075827/156249097-4d0a875c-023b-44fa-835a-06be2a5eec53.jpeg)

![3](https://user-images.githubusercontent.com/65075827/156249104-e83e76dc-46ee-4f60-989a-7049a5d62be8.jpeg)

![5](https://user-images.githubusercontent.com/65075827/156249124-8a56f837-c608-48a6-9b35-71f320c6d190.jpeg)

![6](https://user-images.githubusercontent.com/65075827/156249136-cae90fe7-0c22-4b34-9933-9a1f31690b8d.jpeg)

Resnet 50 Model-
![8](https://user-images.githubusercontent.com/65075827/156249383-86f6a93f-cf83-4220-9a01-e91beb6422f8.jpg)

![7](https://user-images.githubusercontent.com/65075827/156249160-e028a4c7-0343-45b9-a793-b74f67a7da30.jpg)

