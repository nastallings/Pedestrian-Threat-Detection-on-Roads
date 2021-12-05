# Predicting-Pedestrian-Threats
Predicting Pedestrian Threats

This is our final submission for sensor fusion and perception. The code can be found here: https://github.com/nastallings/Pedestrian-Threat-Detection-on-Roads

The code is broken down into multiple files. Road_Detector.py deals with detecting roads and will be run automatically. Pedestrian_Detector.py deals with using CV2's pedestrian detector to locate people within the frames. This will be run automatically. main.py initializes the ring-buffer data structure and loads a video clip that gets spliced frame by frame. Once spliced, it performs normal pedestrian detection. If a pedestrian is detected, it runs road detection. It then performs threat detection on the frame. Depending on the threat detected, it puts a color filter over the video frame. This is the file that needs to be run to demonstrate the project. To get the video to play, you will have to hit or hold space to progress through the frames. 
