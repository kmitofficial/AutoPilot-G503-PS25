
# AutoPilot-G503-PS25

A consolidated repository containing multiple autonomous driving, rover simulation, CNN-based recognition, and Vision-Language-Model (VLM) integrated modules developed for PS-25.

The repo includes experiments across Webots, Gemini, Drive-LM, location-based modules, and CNN simulations.

---

## 🗂️ Repository Structure

```

/
├── Audacity Simulator(CNN)/
│   └── CNN-based audio/gesture simulation notebooks and scripts
│
├── Autonomous Vehicle/
│   └── Modules for autonomous vehicle behaviour, sensors, and control logic
│
├── Gemini_with_waypoints/
│   └── Gemini simulation setup with waypoint navigation
│
├── Webots Simulator with Gemini/
│   └── Webots simulation integrated with Gemini for autonomous vehicle testing
│
├── drivelm/
│   └── Drive LM experiments: longitudinal/lateral control logic, PID/LQR style control
│
├── location/
│   └── Scripts and data for GPS/location/mapping simulation
│
└── rover_sim/
└── Rover simulation environment, control logic, and navigation modules

````

Each folder contains independent experiments. Most modules run in VsCode and colab, with supporting Python scripts.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x  
- Jupyter Notebook / Jupyter Lab  
- For simulations: Webots, Gemini or equivalent simulators  
- GPU recommended for CNN experiments (optional)

### Installation

Clone the repository:
```bash
git clone https://github.com/kmitofficial/AutoPilot-G503-PS25.git
cd AutoPilot-G503-PS25
````

(Optional) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

Install dependencies (if available):

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install missing imports based on individual notebooks.

---

## ▶️ Running Modules

### CNN Simulator (Audacity Simulator)

```bash
cd "Audacity Simulator(CNN)"
jupyter notebook
```

Open the relevant `.ipynb` file and run cell-by-cell.

### Autonomous Vehicle / Webots / Gemini

* Open the simulator (Webots/Gemini).
* Load the world/config from the respective folder.
* Run the Python controller script or follow instructions inside that folder.

### Rover Simulation

```bash
cd rover_sim
jupyter notebook
```

Or run any Python control script inside the folder.

---

## 🎯 Project Goals

* Build CNN-based audio/gesture recognition modules.
* Develop autonomous vehicle logic and simulated behaviour.
* Integrate Webots + Gemini simulations.
* Explore VLM-based decision making for simulation agents.
* Implement rover control, navigation, and environmental interactions.
* Provide extendable testbeds for PS-25 research and demonstration.

---

## 📂 Module Overviews

### **Audacity Simulator (CNN)**

* CNN models for audio/gesture classification
* Data preprocessing notebooks
* Training + Evaluation scripts

### **Autonomous Vehicle**

* Code for lane following
* Sensor simulation
* Motion planning experiments

### **Gemini_with_waypoints**

* Gemini world setup with waypoint arrays
* Tests for waypoint-based navigation logic

### **Webots Simulator with Gemini**

* Webots world files
* Python controller scripts
* Integrated Gemini-based decision layer

### **drivelm**

* Experiments for Drive-LM control
* Longitudinal & lateral control models
* PID/LQR-like tuning notebooks

### **location**

* GPS, latitude-longitude utilities
* Map/coordinate transformation helpers
* Scripts for simulated location input

### **rover_sim**

* Rover movement logic
* Terrain & environment simulation
* Path planning and behaviour testing

---

## 🧪 Testing & Outputs

* CNN: accuracy curves, confusion matrices, audio classification results
* Vehicle & rover simulations: behaviour validation through virtual environments
* Waypoints: trajectory tests in controlled scenes
* You may create a `/results` folder in each module to store logs, models, screenshots, and videos

---

## 📞 Contact

For issues or contributions, use the **GitHub Issues** tab or contact the project team.

---

## 👥 Contributors
---
Sarayu – 23BD1A6744  
Karthik – 23BD1A6704  
Melankita – 23BD1A6738 
Rithwika Reddy – 23BD1A6749 
Surya Akshaya – 23BD1A6723

Mentor: Sai Krishna Sir

---

---

## 📜 License

If you want to include a license, consider MIT:

```
MIT License
Copyright (c) 2025 
Permission is hereby granted, free of charge, to any person obtaining a copy …
```

---

## 📝 Future Enhancements

* RL-based autonomous agent
* Real-time sensor fusion
* Integration of real rover hardware inputs
* Enhanced waypoint planning
* Better simulation-to-real transfer models

---


