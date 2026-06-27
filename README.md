### Real-Time Emotion Detector CNN 🧠📸

A deep learning application that identifies human facial expressions in real-time using a live camera feed. Powered by a Convolutional Neural Network (CNN) built with Python and TensorFlow/Keras, this system captures video frames, localizes faces, and maps expressions to emotional states using the FER-2013 benchmark dataset.

* * *

### ✨ Features

*   **Live Video Analytics**: Connects directly to a local webcam using OpenCV to stream frames and overlay live classification tracking boundaries.
*   **Deep Learning Framework**: Custom Convolutional Neural Network (CNN) designed for facial feature extraction.
*   **FER-2013 Dataset Core**: Trained to classify the 7 primary human universal emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
*   **Pre-trained Architecture**: Includes a compiled model weights configuration (`emotion_model.h5`) to facilitate immediate out-of-the-box local camera testing.
*   **Automated Data Fetching**: Script integration utilizing KaggleHub to seamlessly pull down project training dependencies without leaving the terminal.

* * *

### 🛠️ Built With

*   **Python (100.0%)**: Core project runtime language.
*   **TensorFlow / Keras**: Used for layer definition, pipeline handling, and neural network compilation.
*   **OpenCV**: Drives real-time webcam rendering, video processing loops, bounding boxes, and label rendering.
*   **Kaglehub**: Automates dataset challenge infrastructure acquisition directly from Kaggle platforms.

* * *

### 🚀 Getting Started

### Prerequisites

Install Python 3.x along with the baseline libraries required to handle the deep learning weights and camera inputs:

bash

    pip install tensorflow keras kagglehub opencv-python numpy
    

Use code with caution.

### Installation & Execution

1.  **Clone the repository:**
    
    bash
    
        git clone https://github.com
        
    
    Use code with caution.
    
2.  **Navigate to the directory:**
    
    bash
    
        cd Emotion_Detector_CNN
        
    
    Use code with caution.
    
3.  **Fetch the FER-2013 Dataset Archive:**  
    Run the utility script to map and unpack your training directory assets:
    
    bash
    
        python "import kagglehub.py"
        
    
    Use code with caution.
    
4.  **Retrain the Classifier (Optional):**  
    If you wish to configure layers or run custom training iterations using the raw image data:
    
    bash
    
        python train.py
        
    
    Use code with caution.
    
5.  **Launch the Live Video Feedback:**  
    Boot up your webcam system to monitor bounding boxes and predict your emotions in real-time:
    
    bash
    
        python emotion_detector.py
        
    
    Use code with caution.
    
    *Press `q` on your keyboard while focusing the video window to safely close the video pipeline.*

* * *

### 📁 Project Structure

text

    Emotion_Detector_CNN/
    │
    ├── train.py              # Handles network compilation, data splits, and model training cycles
    ├── emotion_detector.py   # Main live camera utility leveraging OpenCV bounding overlays
    ├── emotion_model.h5      # Pre-trained deep learning binary containing frozen weight parameters
    └── import kagglehub.py   # Script automating localized FER-2013 repository asset downlinks
    

Use code with caution.

* * *

### 🔮 Future Roadmap

*   **Confidence Tracker Bars**: Implement graphic charts tracking accuracy scores side-by-side during active streaming.
*   **Lightweight Backbone Integration**: Adapt architecture parameters to handle MobileNet frames for micro-computing hardware setups.
*   **Web UI Dashboard**: Integrate a custom front-end interface using Streamlit or Flask configurations.

* * *

### 🤝 Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

* * *

### 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
