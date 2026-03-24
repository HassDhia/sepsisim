# sepsisim

**Gymnasium environments for reinforcement learning in sepsis management**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-136%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/sepsisim.svg)](https://pypi.org/project/sepsisim/)

---

SepsiSim provides three Gymnasium-compatible reinforcement learning environments for sepsis management in the intensive care unit. Built on established physiological models (Reynolds et al. 2006 inflammation dynamics, cardiovascular fluid response, lactate kinetics), these environments enable researchers to develop and benchmark RL algorithms for critical care decision-making without risking patient safety.

## Installation

```bash
pip install sepsisim              # Core (numpy, scipy, gymnasium)
pip install sepsisim[train]       # + SB3, PyTorch for RL training
pip install sepsisim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/sepsisim.git
cd sepsisim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import sepsisim

env = gym.make("sepsisim/FluidResuscitation-v0")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environments

| Environment | Paradigm | Observation | Action | Key Challenge |
|---|---|---|---|---|
| `FluidResuscitation-v0` | Continuous | 7D (MAP, lactate, UO, damage, volume, bacteria, SOFA) | 1D (fluid mL) | Restore MAP while avoiding fluid overload |
| `VasopressorTitration-v0` | Continuous | 8D (+ vaso dose, hours on vaso) | 1D (dose change) | Maintain MAP without excessive vasoconstriction |
| `SepsisManagement-v0` | Multi-action | 10D (+ abx status, time) | 3D (fluid, vaso, abx) | Balance fluids, vasopressors, and antibiotic timing |

## Architecture

SepsiSim combines four physiological models into configurable Gymnasium environments:

- **Inflammation Model** (Reynolds et al. 2006): 4-ODE system tracking bacteria, pro-/anti-inflammatory response, and tissue damage
- **Cardiovascular Model**: Lumped-parameter hemodynamic response to fluids and vasopressors with Frank-Starling saturation
- **Lactate Kinetics**: Single-compartment model linking tissue hypoperfusion to lactate production and clearance
- **SOFA Scoring**: Simplified 3-component organ failure assessment (cardiovascular, renal, metabolic)

Each environment supports three difficulty tiers (easy/medium/hard) via initial bacterial load, enabling curriculum learning and systematic benchmarking.

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/sepsisim/blob/main/paper/sepsisim.pdf)

## Citation

If you use sepsisim in your research, please cite:

```bibtex
@software{dhia2026sepsisim,
  author = {Dhia, Hass},
  title = {SepsiSim: Gymnasium Environments for Reinforcement Learning in Sepsis Management},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/sepsisim}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia -- Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
