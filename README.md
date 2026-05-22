# Multi-Agent Path Planning for Environmental Monitoring using SAC

This repository implements a decentralized, autonomous path-planning system for a fleet of mobile agents designed to monitor complex environmental variables (e.g., water quality in lakes or pollution in ports). The system utilizes **Soft Actor-Critic (SAC)**, a state-of-the-art Deep Reinforcement Learning algorithm, to maximize area coverage and locate high-value "peaks" within the environment while autonomously avoiding obstacles.

## 🛠️ Key Features

* **Custom Gymnasium Environment**: A multi-agent simulation framework that supports real-world map data (CSV) with configurable agent sensing and movement physics.
* **Deep Reinforcement Learning**: Implementation of SAC with **Twin Q-Learning** and **Automatic Entropy Tuning** to ensure stable learning and continuous exploration.
* **Map Reconstruction & Evaluation**: Integrates `LinearNDInterpolator` to reconstruct full environmental maps from sparse agent observations, allowing for quantitative performance tracking via **MAE** and **$R^2$** metrics.
* **Modular Architecture**: Refactored from research code into a clean, production-ready directory structure following PEP 8 standards.

##  Project Structure

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

### 2. Training the Fleet

Start the training process with the default configuration. This will log rewards and network loss to TensorBoard:

```bash
python -m src.main

```

### Monitoring Metric Dashboards

To monitor network loss convergences and tracking metrics live, point TensorBoard to your active log folder:

```bash
tensorboard --logdir runs_tuning

```

## 📊 Results and Visualization

The system periodically evaluates the agent's performance by comparing the Ground Truth map against the reconstructed map sampled by the agents.

### Quantitative Performance

* **Reconstruction Accuracy**: Measured via $R^2$ score and Mean Absolute Error (MAE) during deterministic evaluation episodes.
* **Coverage Efficiency**: Tracks the percentage of unique traversable cells visited per time step.

