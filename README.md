
# Awesome TensorFlow Lite [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Twitter](https://img.shields.io/badge/Twitter-%40margaretmz-blue)](https://twitter.com/margaretmz)

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
    - [Segmentation](#segmentation)
    - [Style Transfer](#style-transfer)
    - [Generative](#generative)
    - [Post estimation](#post-estimation)
    - [Other](#other)
  - [Text](#text)
  - [Speech](#speech)
  - [Recommendation](#recommendation)
  - [Game](#game)
- [Model zoo](#model-zoo)
  - [TensorFlow Lite models](#tensorflow-lite-models)
  - [TensorFlow models](#tensorflow-models)
- [Ideas and Inspiration](#ideas-and-inspiration)
- [ML Kit examples](#ml-kit-examples)
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
* [Model Metadata](https://www.tensorflow.org/lite/convert/metadata) - Provides a standard for model descriptions which also enables [Code Gen and Android Studio ML Model Binding](https://www.tensorflow.org/lite/inference_with_metadata/codegen).

## Models with samples
Here are the TensorFlow Lite models with app / device implementations, and references.
Note: pretrained TensorFlow Lite models from MediaPipe are included, which you can implement with or without MediaPipe.

### Computer vision

#### Classification

| Task                            | Model                                                                                                                                                             | App \| Reference                                                                                                                                                                                                                                                                                                                                                                                                       | Source             |
| ------------------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       | -------------------|
| Classification                  | MobileNetV1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip))                          | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) \| [Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) \| [Overview](https://www.tensorflow.org/lite/models/image_classification/overview) | tensorflow.org     |
| Classification                  | MobileNetV2                                                                                                                                                       | Recognize Flowers on Android [Codelab](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0) \| [Android](https://github.com/tensorflow/examples/tree/master/lite/codelabs/flower_classification/android)                                                                                                                                                                   | TensorFlow team    |
| Classification                  | MobileNetV2                                                                                                                                                       | Skin Lesion Detection [Android](https://github.com/AakashKumarNain/skin_cancer_detection/tree/master/demo)                                                                                                                                                                                                                                                                                                             | Community          |
| Classification                  | MobileNetV2                                                                                                                                                       | American Sign Language Detection \| [Colab Notebook](https://colab.research.google.com/drive/1xsunX7Qj_XWBZwcZLyjsKBg4RI0DNo2-?usp=sharing) \| [Android](https://github.com/sayannath/American-Sign-Language-Detection)                                                                                                                                                                                                                                                                                                       | Community          |
| Classification                  | CNN + Quantisation Aware Training                                                                                                                                                       | Stone Paper Scissor Detection [Colab Notebook](https://colab.research.google.com/drive/1Wdso2N_76E8Xxniqd4C6T1sV5BuhKN1o?usp=sharing) \| [Flutter](https://github.com/sayannath/American-Sign-Language-Detection)                                                                                                                                                                                                                                                                                                            | Community          |
| Classification                  | EfficientNet-Lite0 ([download](https://github.com/margaretmz/icon-classifier/blob/master/ml-code/icons-50.tflite))                                                | Icon Classifier [Colab & Android](https://github.com/margaretmz/icon-classifier) \| [tutorial 1](https://medium.com/swlh/icon-classifier-with-tflite-model-maker-9263c0021f72) \| [tutorial 2](https://medium.com/@margaretmz/icon-classifier-android-app-1fc0b727f761)                                                                                                                                                | Community          |

#### Detection
| Task | Model | App \| Reference | Source |
| -|-|-|-|
| Object detection                | Quantized COCO SSD MobileNet v1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios) \| [Overview](https://www.tensorflow.org/lite/models/object_detection/overview#starter_model)                                                                                                                     | tensorflow.org     |
| Object detection                | YOLO                                                                                                                                                              | [Flutter](https://blog.francium.tech/real-time-object-detection-on-mobile-with-flutter-tensorflow-lite-and-yolo-android-part-a0042c9b62c6) \| [Paper](https://arxiv.org/abs/1506.02640)    | Community          |
| Object detection                             | [YOLOv5](https://tfhub.dev/neso613/lite-model/yolo-v5-tflite/tflite_model/1)     | [Yolov5 Inference ](https://github.com/neso613/yolo-v5-tflite-model)  | Community   |
| Object detection                | MobileNetV2 SSD ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite))                                    | [Reference](https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_saved_model/README.md)                                                                                                                                                                                                                                                                                                   | MediaPipe          |
| Object detection                | MobileDet ([Paper](https://arxiv.org/abs/2004.14525))                                    | [Blog post (includes the TFLite conversion process)](https://sayak.dev/mobiledet-optimization/)                                                                                                                                                                                                                                                                                                   | MobileDet is from University of Wisconsin-Madison and Google and the blog post is from the Community          |
| License Plate detection         | SSD MobileNet [(download)](https://github.com/ariG23498/Flutter-License/blob/master/assets/detect.tflite)                                                         | [Flutter](https://github.com/ariG23498/Flutter-License)                                                                                                                                                                                                                                                                                                                                                                | Community          |
| Face detection                  | BlazeFace ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite))                                              | [Paper](https://sites.google.com/corp/view/perception-cv4arvr/blazeface)                                                                                                                                                                                                                                                                                                                                               | MediaPipe          |
| Face Authentication                  | [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)                                            | [Flutter](https://github.com/sayannath/Face-Authentication-App)                                                                                                                                                                                                                                                                                                                                               | Community          |
| Hand detection & tracking       | Palm detection & hand landmarks ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models#hand-detection-and-tracking))                        | [Blog post](https://mediapipe.page.link/handgoogleaiblog) \| [Model card](https://mediapipe.page.link/handmc) \|  [Android](https://github.com/supremetech/mediapipe-demo-hand-detection)                                                                                                                                                                                                                                                                                                         | MediaPipe & Community         |

#### Other
| Task | Model | App \| Reference | Source |
| -|-|-|-|
| Low-light image enhancement   | [Models on TF Hub](https://tfhub.dev/sayakpaul/mirnet-fixed/1)                                                     | [Project repo](https://github.com/sayakpaul/MIRNet-TFLite)  \| [Original Paper](https://arxiv.org/pdf/2003.06792v2.pdf) \| [Flutter](https://github.com/sayannath/MIRNet-Flutter)|                                                                                                                           | Community          |
| OCR                             |[Models on TF Hub](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/dr/2)     | [Project Repository](https://github.com/tulasiram58827/ocr_tflite)  | Community


### Text
| Task                | Model                                                                                                                           | Sample apps                                                                                                                                                                                                                                       | Source             |
| ------------------- |---------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Question & Answer   | DistilBERT                                                                                                                      | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/bert)                                                                                                                                                            | Hugging Face       |
| Text Generation     | GPT-2 / DistilGPT2                                                                                                              | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2)                                                                                                                                                            | Hugging Face       |
| Text Classification | [Download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) \|[iOS](https://github.com/khurram18/TextClassafication) \| [Flutter](https://github.com/am15h/tflite_flutter_plugin/tree/master/example) | tf.org & Community |
| Text Detection                  | CRAFT Text Detector ([Paper](https://arxiv.org/pdf/1904.01941))                          |[Download](https://github.com/tulasiram58827/craft_tflite/blob/main/models/craft_float_800.tflite?raw=true) \| [Project Repository](https://github.com/tulasiram58827/craft_tflite/)  \| [Blog1-Conversion to TFLite](https://tulasi.dev/craft-in-tflite) \| [Blog2-EAST vs CRAFT](https://sayak.dev/optimizing-text-detectors/) \| [Models on TF Hub](https://tfhub.dev/tulasiram58827/lite-model/craft-text-detector/dr/1)   \| Android (Coming Soon)                                 | Community          |
| Text Detection                  | EAST Text Detector ([Paper](https://arxiv.org/abs/1704.03155))                          |[Models on TF Hub](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/dr/1) \| [Conversion and Inference Notebook](https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/EAST_TFLite.ipynb)  | Community          |

### Speech
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| Speech Recognition | DeepSpeech                         | [Reference](https://github.com/mozilla/DeepSpeech/tree/master/native_client/java)     | Mozilla      |
| Speech Recognition | CONFORMER                          | [Inference](https://github.com/neso613/ASR_TFLite)  [Android](https://github.com/windmaple/tflite-asr) | Community |
| Speech Synthesis   | Tacotron-2, FastSpeech2, MB-Melgan | [Android](https://github.com/TensorSpeech/TensorflowTTS/tree/master/examples/android) | TensorSpeech |
| Speech Synthesis(TTS)   | Tacotron2, FastSpeech2, MelGAN, MB-MelGAN, HiFi-GAN, Parallel WaveGAN | [Inference Notebook](https://github.com/tulasiram58827/TTS_TFLite/blob/main/End_to_End_TTS.ipynb)      \| [Project Repository](https://github.com/tulasiram58827/TTS_TFLite/)  | Community  |

### Recommendation
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| On-device Recommendation | [Dual-Encoder](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)                 | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android) \| [iOS](https://github.com/zhuzilin/on-device_recommendation_tflite) \| [Reference](https://blog.tensorflow.org/2020/09/introduction-to-tflite-on-device-recommendation.html)     | tf.org & Community      |

### Game
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| Game agent | Reinforcement learning                 | [Flutter](https://github.com/windmaple/planestrike-flutter) \| [Tutorial](https://windmaple.medium.com/)     | Community      |



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

## ML Kit examples
[ML Kit](https://developers.google.com/ml-kit) is a mobile SDK that brings Google's ML expertise to mobile developers.
* 2019-10-01 [ML Kit Translate demo](https://codelabs.developers.google.com/codelabs/mlkit-android-translate/#0) - A tutorial with material design [Android](https://github.com/googlecodelabs/mlkit-android/tree/master/translate) (Kotlin) sample - recognize, identify Language and translate text from live camera with ML Kit for Firebase.
* 2019-03-13 [Computer Vision with ML Kit - Flutter In Focus](https://youtu.be/ymyYUCrJnxU).
* 2019-02-09 [Flutter + MLKit: Business Card Mail Extractor](https://medium.com/flutter-community/flutter-mlkit-8039ec66b6a)  - A blog post with a [Flutter](https://github.com/DaemonLoki/Business-Card-Mail-Extractor) sample code.
* 2019-02-08 [From TensorFlow to ML Kit: Power your Android application with machine learning](https://speakerdeck.com/jinqian/from-tensorflow-to-ml-kit-power-your-android-application-with-machine-learning) - A talk with [Android](https://github.com/xebia-france/magritte) (Kotlin) sample code.
* 2018-08-07 [Building a Custom Machine Learning Model on Android with TensorFlow Lite](https://medium.com/over-engineering/building-a-custom-machine-learning-model-on-android-with-tensorflow-lite-26447e53abf2).
* 2018-07-20 [ML Kit and Face Detection in Flutter](https://flatteredwithflutter.com/ml-kit-and-face-detection-in-flutter/).
* 2018-07-27 [ML Kit on Android 4: Landmark Detection](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-landmark-detection-part-four-5e86b8deac3a).
* 2018-07-28 [ML Kit on Android 3: Barcode Scanning](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-barcode-scanning-part-three-cc6f5921a108).
* 2018-05-31 [ML Kit on Android 2: Face Detection](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-face-detection-part-two-de7e307c52e0).
* 2018-05-22 [ML Kit on Android 1: Intro](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-introducing-mlkit-part-one-98fcfedbeee0).

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
