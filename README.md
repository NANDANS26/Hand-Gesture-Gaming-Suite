## 🚀 The Ultimate Hand Gesture Gaming Suite (Python + CV)

### **A Fusion of Pygame and MediaPipe for Next-Level Human-Computer Interaction**

This repository showcases five unique games, each driven entirely by **real-time hand gestures** captured via a webcam. Built on Python, this project combines the power of **OpenCV** and **MediaPipe** for robust computer vision with the fun of **Pygame** for high-performance game rendering.

---

## 🎮 Game Showcase & Controls Breakdown

| Game Name | Control Mechanism | Key Features | Tech Focus |
| :---: | :--- | :--- | :--- |
| **1. 🐍 CV-Snake** | **Wrist-Vector Direction:** Directional control by the vector from the **Wrist** to the **Average Fingertip Position**. | Real-time speed scaling, pulsing food animation, precise **Vector** control. | Computer Vision Stability |
| **2. ⚔️ Gesture RPS** | **Static Pose:** Closed Fist (✊) $\rightarrow$ **Rock**; Open Hand (✋) $\rightarrow$ **Paper**; Two Fingers (✌️) $\rightarrow$ **Scissors**. | **Professional OpenCV UI**, adaptive CPU opponent AI, animated countdown and score tracking. | Pose Classification, State Management |
| **3. 🍉 Fruit Ninja** | **Motion Swipe:** Fast hand movement/swipe of the **Index Finger Tip** to register a cut. | Line-circle intersection slice detection, **Combo System**, parabolic trajectories, **Screen Shake** on bomb hit. | Motion Tracking, Physics Simulation |
| **4. 🏎️ Perfect Racer** | **Dual-Input:** 1️⃣ **Hand Tilt** for Steering; 2️⃣ **Thumb Up** (Gas); **Pinky Up** (Brake). | **Dual-Input Gesture System**, visual **Boost/Drift** effects, custom speed & damage physics. | Multi-Class Gesture Recognition |
| **5. 🌀 Maze Solver** | **Switchable Modes:** 1️⃣ **Palm Tilt** $\rightarrow$ Movement; 2️⃣ **Index/Pinky Point** $\rightarrow$ Movement. | **Switchable Control Modes**, DFS-based maze generation, **300ms Move Delay** for stability. | Directional Estimation, Control Smoothing |

---

## ✨ Next-Level Visualization & Data

The project prioritizes visual feedback and clean design to enhance the user experience.

### 📊 Gesture Debugging & Visualization

| Game | Visual Feedback in Camera Window | Gesture Stability Mechanism |
| :--- | :--- | :--- |
| **CV-Snake** | Green arrow showing the **Wrist-to-Fingertip** directional vector. Blue/Red circles on the average fingertip and wrist centers. | **$\Delta x/\Delta y$ Threshold** (Sensitivity setting is adjustable with UP/DOWN keys). |
| **Perfect Racer**| Animated **Steering Wheel** visualization that rotates with hand tilt. Illuminated **GAS/BRAKE** pedals based on finger position. | **Majority-Vote Smoothing** over the last 5 frames of input using `deque`. |
| **Maze Solver** | Displays the **Current Method** (Tilt/Fingers) and the detected direction with a large center arrow. | **Stability Counter** (2 consecutive frames of the same input required) + **Move Cooldown**. |

### 🛠️ Key Mappings (Across Games)

| Key | Function | Applicable Games |
| :--- | :--- | :--- |
| **Q / ESC** | Quit Game / Close Camera Window | All Games |
| **R** | Restart Game / Play Again | Racer, RPS, Fruit Ninja, Snake (SPACE) |
| **SPACE** | Restart Game | Snake |
| **T** | Toggle Gesture Control Method | Maze Solver |

---

## 💻 Running Guidelines

### Project Structure (Reflecting Image)

The file organization is neat and easy to navigate:

GEUSTUREGAMES/ 
├── fruit/ 
│ └── main.py 
├── maze/ 
│ └── main.py 
├── racer/ 
│ └── main.py 
├── rock/ 
│ └── main.py 
├── snake/ 
│ └── main.py 
├── README.md 
└── requirements.txt


### Prerequisites

You need **Python 3.x** installed. The core dependencies are managed via `requirements.txt`.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Hand-Gesture-Gaming-Suite.git](https://github.com/YourUsername/Hand-Gesture-Gaming-Suite.git)
    cd Hand-Gesture-Gaming-Suite
    ```

2.  **Install dependencies:**
    *(Note: Sound files for `racer/main.py` are not included but are gracefully handled. The necessary libraries will be installed.)*

    ```bash
    # Optionally create and activate a virtual environment first
    pip install -r requirements.txt 
    # The requirements.txt should contain: pygame opencv-python mediapipe numpy
    ```

### Execution

Each game runs independently and requires access to your computer's **webcam** (Camera Index 0).

| Game | Command | Notes |
| :--- | :--- | :--- |
| **CV-Snake** | `python snake/main.py` | Press **UP/DOWN** keys to adjust gesture sensitivity. |
| **Gesture RPS** | `python rock/main.py` | Press **'S'** to start the match in the OpenCV window. |
| **Fruit Ninja** | `python fruit/main.py` | Use quick, sharp movements of your hand to slice. |
| **Perfect Racer**| `python racer/main.py` | Keep your hand steady in the frame to maintain speed (cruise). |
| **Maze Solver** | `python maze/main.py` | Use the **R** key to immediately reset and generate a new maze. |

---

## 🔬 Core Technology Stack

### MediaPipe Landmark Analysis

All controls are derived from analyzing the 21 key points provided by the MediaPipe Hands model.


* **RPS:** Relies on the $y$-axis comparison between the fingertip landmarks ($\#8, \#12, \#16, \#20$) and their corresponding knuckles.
* **Snake:** Relies on the displacement vector between the Wrist ($\#0$) and the average of the four non-thumb fingertips.
* **Racer:** Uses complex analysis involving the Wrist ($\#0$), Middle MCP ($\#9$), Thumb CMC ($\#1$), and Pinky/Index MCPs ($\#5, \#17$) to calculate tilt and acceleration.

---

## 👤 Author

* **Nandan S** - [Your LinkedIn Profile Link]
* **Project Link** - [Link to this GitHub Repository]

*A ⭐ star on this repo would be highly appreciated! Thank you for checking out the proj
