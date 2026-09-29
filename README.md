# AI-Based Sign Language to Text Translation System

An AI-powered sign language recognition system that aims to translate sign language into readable text in real time.

## 📌 Project Overview

Communication can be difficult between people who use sign language and people who do not understand it.

This project aims to develop an AI-based system that can recognize sign language from camera input and convert the recognized signs into text.

### Example

Sign Language
      ↓
     AI
      ↓
Recognized Signs
      ↓
Sentence
      ↓
"I want to go home."


The long-term goal is to develop this system into a mobile application that can assist communication between sign language users and people who do not know sign language.

## 🎯 Objectives

* Detect sign language using a camera.
* Extract useful hand and body movements.
* Recognize individual signs using machine learning/deep learning.
* Combine recognized signs into meaningful text.
* Support a growing vocabulary.
* Develop the system toward real-time translation.
* Prepare the system for future mobile application development.

## 🧠 Planned AI Pipeline

Camera / Video
      ↓
Frame Processing
      ↓
Hand & Body Landmark Detection
      ↓
Feature Extraction
      ↓
Sign Recognition Model
      ↓
Recognized Signs
      ↓
Sentence Construction
      ↓
Text Output

## 🚀 Development Stages

### Stage 1 - Basic Sign Recognition

Start with a small number of sign classes and build a working recognition model.

### Stage 2 - Vocabulary Expansion

Increase the number of recognizable signs using a larger dataset.

### Stage 3 - Continuous Sign Recognition

Move from recognizing individual images to recognizing sequences of signs from video.

### Stage 4 - Sentence Construction

Combine recognized signs into readable text.

### Stage 5 - Real-Time Translation

Develop a real-time camera-based translation system.

### Stage 6 - Mobile Application

Explore deployment of the trained AI model in a mobile application.

## 📊 Dataset

The project will use publicly available sign language datasets during development and may later incorporate a custom dataset.

The initial research and development will focus on American Sign Language (ASL) datasets because of their availability for machine-learning research.

Potential datasets include:

* WLASL
* Google Isolated Sign Language Recognition Dataset
* Other suitable sign-language datasets

Dataset licenses and usage requirements will be checked before using any dataset.

## 🛠️ Technologies

### Programming

* Python

### Computer Vision

* OpenCV
* MediaPipe

### Machine Learning / Deep Learning

* Scikit-learn
* TensorFlow / PyTorch

### Development

* Visual Studio Code
* Git
* GitHub

## 📁 Planned Project Structure

AI-Sign-Language-to-Text/
│
├── dataset/
├── data/
├── models/
├── src/
│
├── notebooks/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore

## 🔮 Future Improvements

* Support a larger vocabulary.
* Recognize continuous signing.
* Improve sentence construction.
* Add text-to-speech.
* Improve recognition under different lighting conditions.
* Support multiple signers.
* Develop a mobile application.
* Explore support for Sri Lankan Sign Language.

## ⚠️ Current Status

This project is under active development.

The initial version focuses on building the sign recognition pipeline. More advanced continuous sentence translation will be developed in later stages.

## 👩‍💻 Project Goal

The ultimate goal is to create an accessible AI system that can help bridge communication between sign language users and people who do not understand sign language.
