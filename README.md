# Erg.ai: a real-time erging analyzer

## Overview
Inspired by my competitive rowing background, I built a computer vision biomechanics analyzer that can provide real-time video overlays for erg videos.

## Technical Description
The core of this analyzer uses OpenCV and mediapipe pose detection to analyze erg film, with continuous pose data smoothened using exponential moving averages. This data is used to calculate a variety of metrics, such as knee and body angles, stroke rate, and similarity to reference strokes, which are collected from online youtube videos scraped via pytube.
