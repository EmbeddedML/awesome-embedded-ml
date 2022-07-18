
# Awesome TensorFlow Lite [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[TensorFlow Lite](https://www.tensorflow.org/lite) is a set of tools that help convert and optimize TensorFlow models to run on mobile and edge devices. It's currently running on more than 4 billion devices! With TensorFlow 2.x, you can train a model with tf.Keras, easily convert a model to .tflite and deploy it; or you can download a pretrained TensorFlow Lite model from the model zoo.

This is an awesome list of machine learning on embedded devices including models with sample apps, helpful tools and learning resources
* Showcase what the community has built with TensorFlow Lite
* Put all the samples side-by-side for easy reference
* Share knowledge and learning resources

Please submit a PR if you would like to contribute and follow the guidelines [here](CONTRIBUTING.md).

<!-- omit in toc -->
 ## Contents
- [Past announcements:](#past-announcements)
- [Models with samples](#models-with-samples)
  - [Computer vision](#computer-vision)
    - [Classification](#classification)
    - [Detection](#detection)
    - [Generative](#generative)
    - [Other](#other)
  - [Text](#text)
  - [Speech](#speech)
  - [Recommendation](#recommendation)
- [Model zoo](#model-zoo)
  - [TensorFlow Lite models](#tensorflow-lite-models)
  - [TensorFlow models](#tensorflow-models)
- [Ideas and Inspiration](#ideas-and-inspiration)
- [Plugins and SDKs](#plugins-and-sdks)
- [Helpful links](#helpful-links)
- [Learning resources](#learning-resources)
  - [Blog posts](#blog-posts)
  - [Books](#books)
  - [Videos](#videos)
  - [Podcasts](#podcasts)
  - [MOOCs](#moocs)

## Past announcements:
Here are some past feature annoucements of TensorFlow Lite:
* [Announcement of the new converter](https://groups.google.com/a/tensorflow.org/d/msg/tflite/Z_h7706dt8Q/sNrjPj4yGgAJ) - [MLIR](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)-based and enables conversion of new classes of models such as Mask R-CNN and Mobile BERT etc., supports functional control flow and better error handling during conversion. Enabled by default in the nightly builds\.
* [Model Maker](https://www.tensorflow.org/lite/guide/model_maker) - Create your custom [image & text](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker) classification models easily in a few lines of code. See below the Icon Classifier for a tutorial by the community.
* [Model Metadata](https://www.tensorflow.org/lite/convert/metadata) - Provides a standard for model descriptions.

## Models with samples
Here are the TensorFlow Lite models with app / device implementations, and references.
Note: pretrained TensorFlow Lite models from MediaPipe are included, which you can implement with or without MediaPipe.

### Computer vision

#### Classification

| Model | App/Reference | Source |
| -|-|-|
| Classification| MobileNetV1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip))| [Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) \| [Overview](https://www.tensorflow.org/lite/models/image_classification/overview) | tensorflow.org     |

#### Detection
| Model | App/Reference | Source |
| -|-|-|
| Quantized COCO SSD MobileNet v1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)) |  [Overview](https://www.tensorflow.org/lite/models/object_detection/overview#starter_model)                                                                                                                     | tensorflow.org     |

#### Other
| Task | Model | App \| Reference | Source |
| -|-|-|-|
| Low-light image enhancement   | [Models on TF Hub](https://tfhub.dev/sayakpaul/mirnet-fixed/1)                                                     | [Project repo](https://github.com/sayakpaul/MIRNet-TFLite)  \| [Original Paper](https://arxiv.org/pdf/2003.06792v2.pdf) \| [Flutter](https://github.com/sayannath/MIRNet-Flutter)|                                                                                                                           | Community          |
| OCR                             |[Models on TF Hub](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/dr/2)     | [Project Repository](https://github.com/tulasiram58827/ocr_tflite)  | Community


### Speech
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| Speech Recognition | DeepSpeech                         | [Reference](https://github.com/mozilla/DeepSpeech/tree/master/native_client/java)     | Mozilla      |
| Speech Recognition | CONFORMER                          | [Inference](https://github.com/neso613/ASR_TFLite) | Community |
| Speech Synthesis   | Tacotron-2, FastSpeech2, MB-Melgan |  TensorSpeech |
| Speech Synthesis(TTS)   | Tacotron2, FastSpeech2, MelGAN, MB-MelGAN, HiFi-GAN, Parallel WaveGAN | [Inference Notebook](https://github.com/tulasiram58827/TTS_TFLite/blob/main/End_to_End_TTS.ipynb)      \| [Project Repository](https://github.com/tulasiram58827/TTS_TFLite/)  | Community  |

### Recommendation
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| On-device Recommendation | [Dual-Encoder](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)  |  \| [Reference](https://blog.tensorflow.org/2020/09/introduction-to-tflite-on-device-recommendation.html)     | tf.org & Community      |


## Model zoo

### TensorFlow Lite models
These are the TensorFlow Lite models that could be implemented in apps and things:
* [MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md) - Pretrained MobileNet v2 and v3 models.
* TensorFlow Lite models
  * [TensorFlow Lite models](https://www.tensorflow.org/lite/models) - With official Android and iOS examples.
  * [Pretrained models](https://www.tensorflow.org/lite/guide/hosted_models) - Quantized and floating point variants.
  * [TensorFlow Hub](https://tfhub.dev/) - Set "Model format = TFLite" to find TensorFlow Lite models.

### TensorFlow models
These are TensorFlow models that could be converted to .tflite and then implemented in apps and things:
* [TensorFlow models](https://github.com/tensorflow/models/tree/master/official) - Official TensorFlow models.
* [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) - Pre-trained on COCO, KITTI, AVA v2.1, iNaturalist Species datasets.

## Ideas and Inspiration
* [E2E TFLite Tutorials](https://github.com/ml-gde/e2e-tflite-tutorials) - Checkout this repo for sample app ideas and seeking help for your tutorial projects. Once a project gets completed, the links of the TensorFlow Lite model(s), sample code and tutorial will be added to this awesome list.

## Plugins and SDKs
* [Edge Impulse](https://www.edgeimpulse.com/) - Created by [@EdgeImpulse](https://twitter.com/EdgeImpulse) to help you to train TensorFlow Lite models for embedded devices in the cloud.
* [MediaPipe](https://github.com/google/mediapipe) - A cross platform (mobile, desktop and Edge TPUs) AI pipeline by Google AI. (PM [Ming Yong](https://twitter.com/realmgyong)) | [MediaPipe examples](https://mediapipe.readthedocs.io/en/latest/examples.html).
* [Coral Edge TPU](https://coral.ai/) - Edge hardware by Google. [Coral Edge TPU examples](https://coral.ai/examples/).

## Helpful links
* [Netron](https://github.com/lutzroeder/netron) - A tool for visualizing models.
* [Material design guidelines for ML](https://material.io/collections/machine-learning/patterns-for-machine-learning-powered-features.html) - How to design machine learning powered features. A good example: [ML Kit Showcase App](https://github.com/firebase/mlkit-material-android).
* [The People + AI Guide book](https://pair.withgoogle.com/) - Learn how to design human-centered AI products.
* [Adventures in TensorFlow Lite](https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite) - A repository showing non-trivial conversion processes and general explorations in TensorFlow Lite.
* [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

## Learning resources
Interested but not sure how to get started? Here are some learning resources that will help you whether you are a beginner or a practitioner in the field for a while.

### Blog posts
* 2021-11-09 [On-device training in TensorFlow Lite](https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html)
* 2021-09-27 [Optical character recognition with TensorFlow Lite: A new example app](https://blog.tensorflow.org/2021/09/blog.tensorflow.org202109optical-character-recognition.html)
* 2021-06-16 [https://blog.tensorflow.org/2021/06/easier-object-detection-on-mobile-with-tf-lite.html](https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html)
* 2020-12-29 [YOLOv3 to TensorFlow Lite Conversion](https://medium.com/analytics-vidhya/yolov3-to-tensorflow-lite-conversion-4602cec5c239) - By Nitin Tiwari.
* 2020-04-20 [What is new in TensorFlow Lite](https://blog.tensorflow.org/2020/04/whats-new-in-tensorflow-lite-from-devsummit-2020.html) - By Khanh LeViet.
* 2020-04-17 [Optimizing style transfer to run on mobile with TFLite](https://blog.tensorflow.org/2020/04/optimizing-style-transfer-to-run-on-mobile-with-tflite.html) - By Khanh LeViet and Luiz Gustavo Martins.
* 2020-04-14 [How TensorFlow Lite helps you from prototype to product](https://blog.tensorflow.org/2020/04/how-tensorflow-lite-helps-you-from-prototype-to-product.html) -  By Khanh LeViet.
* 2019-11-08 [Getting  Started with ML on MCUs with TensorFlow](https://blog.particle.io/2019/11/08/particle-machine-learning-101/) -  By Brandon Satrom.
* 2019-08-05 [TensorFlow Model Optimization Toolkit — float16 quantization halves model size](https://blog.tensorflow.org/2019/08/tensorflow-model-optimization-toolkit_5.html) - By the TensorFlow team.
* 2018-07-13 [Training and serving a real-time mobile object detector in 30 minutes with Cloud TPUs](https://blog.tensorflow.org/2018/07/training-and-serving-realtime-mobile-object-detector-cloud-tpus.html) - By Sara Robinson, Aakanksha Chowdhery, and Jonathan Huang.
* 2018-06-11 - [Why the Future of Machine Learning is Tiny](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/) - By Pete Warden.

### Books
* 2021-12-01 [AI and Machine Learning On-Device Development](https://learning.oreilly.com/library/view/ai-and-machine/9781098101732/) - By Laurence Moroney
* 2020-10-01 [AI and Machine Learning for Coders](https://learning.oreilly.com/library/view/ai-and-machine/9781492078180/) - By Laurence Moroney
* 2020-04-06 [Mobile Deep Learning with TensorFlow Lite, ML Kit and Flutter](https://www.packtpub.com/product/mobile-deep-learning-with-tensorflow-lite-ml-kit-and-flutter/9781789611212): Build scalable real-world projects to implement end-to-end neural networks on Android and iOS ([GitHub](https://github.com/PacktPublishing/Mobile-Deep-Learning-Projects)) - By Anubhav Singh ([@xprilion](https://github.com/xprilion)) and Rimjhim Bhadani ([@Rimjhim28](https://github.com/Rimjhim28)).
* 2020-03-01 Raspberry Pi for Computer Vision ([Complete Bundle](https://www.pyimagesearch.com/raspberry-pi-for-computer-vision) | [TOC](https://www.pyimagesearch.com/2019/04/05/table-of-contents-raspberry-pi-for-computer-vision/)) - By the PyImageSearch Team: Adrian Rosebrock, David Hoffman, Asbhishek Thanki, Sayak Paul , and David Mcduffee.
* 2019-12-01 [TinyML](http://shop.oreilly.com/product/0636920254508.do) - By Pete Warden and Daniel Situnayake.
* 2019-10-01 [Practical Deep Learning for Cloud, Mobile, and Edge](https://www.practicaldeeplearning.ai/) - By Anirudh Koul, Siddha Ganju, and Meher Kasam.

### Videos
* [Contributing to TensorFlow Lite with Sunit Roy](https://youtu.be/sZayUoWW6nE) (Hacktoberfest 2021)
* [Easy on-device ML from prototype to production](https://youtu.be/ALxWJoh_BHw) (TF Dev Summit 2020).
* [TensorFlow Lite: ML for mobile and IoT devices](https://youtu.be/27Zx-4GOQA8) (TF Dev Summit 2020).
* [Keynote - TensorFlow Lite: ML for mobile and IoT devices](https://youtu.be/zjDGAiLqGk8).
* [TensorFlow Lite: Solution for running ML on-device](https://youtu.be/0SpZy7iouFU).
* [TensorFlow model optimization: Quantization and pruning](https://youtu.be/3JWRVx1OKQQ).
* [Inside TensorFlow: TensorFlow Lite](https://youtu.be/gHN0jDbJz8E).

### MOOCs
* [Introduction to TensorFlow Lite](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190) - Udacity course by Daniel Situnayake (@dansitu), Paige Bailey, and Juan Delgado.
* [Device-based Models with TensorFlow Lite](https://www.coursera.org/learn/device-based-models-tensorflow) - Coursera course by Laurence Moroney.
* [The Future of ML is Tiny and Bright](https://www.edx.org/professional-certificate/harvardx-tiny-machine-learning) - A series of edX courses created by Harvard in collaboration with Google. Instructors - Vijay Janapa Reddi, Laurence Moroney, and Pete Warden.
