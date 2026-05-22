
APP LINK=


# 3-Path-Planning: Multi-Agent Deep Reinforcement Learning for Adaptive Environmental Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Framework-Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![Engine-PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![UI-Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An end-to-end Multi-Agent Reinforcement Learning (MARL) framework and interactive web application designed for autonomous spatial exploration, adaptive monitoring, and real-time environmental reconstruction. Featuring a custom continuous-action Gymnasium environment paired with a modular Soft Actor-Critic (SAC) implementation, the system optimizes path planning for a team of autonomous agents tasked with sampling and reconstructively mapping dynamic geographic regions (such as Ypacarai Lake and Acoruna Port).

---

## 🚀 Key Features

* **Multi-Agent Continuous Control:** Decentralized decision-making utilizing a Gaussian Policy architecture mapping continuous 2D step velocities.
* **Custom Spatial Gymnasium Environment:** Simulates multi-agent traversal on real-world GIS array structures featuring collision handling, obstacle masking, and path-visitation penalties.
* **Soft Actor-Critic (SAC) Core:** Fully implemented from scratch using a double Q-network configuration, target network soft updates, and entropy-regularized policy sampling via the reparameterization trick.
* **Interactive Streamlit Dashboard:** A high-performance dashboard that orchestrates autonomous missions, dynamically extracts GIS CSV matrix dimensions, loads model checkpoints, and showcases live telemetry.
* **On-the-Fly Spatial Interpolation:** Features an evaluation pipeline leveraging `scipy`'s N-dimensional linear interpolation to reconstruct environmental maps from sparse, agent-collected runtime metrics.
* **Production Telemetry:** Integrates TensorBoard logging alongside automated programmatic rendering pipelines to visualize model evaluation maps (`matplotlib`) against actual ground truth datasets.

---

## 📁 Repository Structure

```text
3-PATH-PLANING/
├── data/                          # Geographic spatial reference tables (CSV)
│   ├── acoruna_port.csv
│   ├── ypacarai_lake_58x41.csv
│   └── ypacarai_map.csv
├── runs_tuning/                   # TensorBoard session logs & rendered evaluations
│   └── sac_experiment/
│       ├── events.out.tfevents... # Experiment metric summaries
│       └── final_model.pth        # Saved PyTorch policy weights
├── src/                           # System core architecture package
│   ├── __init__.py
│   ├── agent.py                   # SAC Agent and experience replay systems
│   ├── environment.py             # MultiAgentMonitoring Gymnasium environment
│   ├── main.py                    # Optimization and orchestration loop
│   ├── models.py                  # PyTorch actor & twin-critic neural networks
│   └── utils.py                   # Statistical evaluation & map reconstruction
├── app.py                         # Streamlit Interactive Dashboard UI
├── .gitignore
└── requirements.txt               # System runtime dependencies

```

---

## ⚙️ Core Architecture

### Neural Architectures (`models.py`)

* **Twin-Q Critics:** Double deep networks modeling $Q(s, a)$ to mitigate maximization bias during dynamic value approximation updates.
* **Gaussian Actor:** Generates a continuous parameter distribution bounded via a scaled Hyperbolic Tangent ($\tanh$) layer, natively enabling stable stochastic action exploration.

### Environment Dynamics (`environment.py`)

* **State Spaces:** Continuously normalizes multi-agent global coordinates and localized scalar payload variables.
* **Reward Mapping:** A dense reward structure balancing global coordinate cell coverage optimizations with robust penalization for geographic boundary violations.

### Evaluation & Dashboard Mesh-Matching (`app.py`, `utils.py`)

Uses deterministic validation routines to pass agent paths to a multidimensional interpolation framework (`LinearNDInterpolator`), quantifying mapping efficacy across standard ML markers:

* Mean Absolute Error (MAE)
* Coefficient of Determination ($R^2$ Score)

The Streamlit dashboard isolates the runtime engine loop to prevent layout breakdown, forcing on-the-fly triangulation array updates to visually display agent path exploration alongside ground-truth matrices.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/vivekbiju/SAC-Based-Multi-Agent-Environmental-Mapping.git](https://github.com/vivekbiju/SAC-Based-Multi-Agent-Environmental-Mapping.git)
   cd SAC-Based-Multi-Agent-Environmental-Mapping
cd 3-path-planning

```


2. **Configure a local virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install standard dependencies:**
```bash
pip install -r requirements.txt

```



---

## 🏃 Running the Framework

### Launching the Interactive Web UI Dashboard

To spin up the web interface for real-time inference rendering:

```bash
streamlit run app.py

```

### Launching the Headless Backend Training Loop

To train the neural networks from scratch via terminal execution:

```bash
python -m src.main

```

### Monitoring Metric Dashboards

To monitor network loss convergences and tracking metrics live, point TensorBoard to your active log folder:

```bash
tensorboard --logdir=runs_tuning/sac_experiment

```

