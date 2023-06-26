This is our final year project for the Human Activity Recognition(HAR) application.

One of the vast areas of study in the field of deep learning is human activity recognition (HAR). It involves automatically detecting human activity from sensor data, such as that from magnetometers, accelerometers, and gyroscopes. Healthcare, sports, entertainment, security, smart homes, and automated systems have all seen extensive usage of HAR. It should be cost-effective and use little resources and computing power for real-time HAR applications.  
  
The primary goal of this paper is to train the model on variable width (number of channels). Additionally, it includes concepts like using the Random Sampling approach rather than fixed sampling and switching the Normal Convolutional Layer with the Lower Triangular Convolutional Layer. The model is developed after preprocessing the sensor data that was used as input. 
  
The produced model is additionally evaluated for usability and real-world practicality on a variety of IoT devices. By default, the model is created using a 8 switch network (i.e., 8 random widths between 0 and 1). It is also trained using a 4-switch network and comparison is being made. The model is tested on a Raspberry Pi 3B and a Node MCU model ESP8266 in this research report. Wearable gloves with an accelerometer sensor are tested. A website that measures a person's body temperature and heart rate is also created for ease of use.

# Dataset links

UCI HAR Dataset - https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

WISDM Dataset   - https://www.cis.fordham.edu/wisdm/dataset.php

Opportunity Dataset - https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

Unimib SHAR Dataset - https://www.sal.disco.unimib.it/technologies/unimib-shar/

PAMAP2 Dataset - https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
