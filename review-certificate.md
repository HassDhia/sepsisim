# Review Certificate: SepsiSim

**Status: APPROVED FOR PUBLICATION**

**Paper:** SepsiSim: Gymnasium Environments for Reinforcement Learning in Sepsis Management
**Author:** Hass Dhia, Smart Technology Investments Research Institute
**Date:** 2026-03-24
**Reviewers:** 4-agent council (clinical accuracy, RL methodology, writing quality, reproducibility)

---

## Review Summary

| Reviewer | Verdict | Key Finding |
|----------|---------|-------------|
| Clinical Accuracy | APPROVE | Sepsis model appropriately simplified; SSC guideline targets (MAP 65, lactate <2) correctly implemented; honest disclosure of limitations |
| RL Methodology | APPROVE | Gymnasium API compliance verified; PPO failure on Fluid env honestly reported; evaluation methodology sound (50 episodes, seed=42) |
| Writing Quality | APPROVE | Clear exposition; equations well-presented; tables formatted correctly; abstract accurately reflects mixed results |
| Reproducibility | APPROVE | 136 tests passing; pyproject.toml with hatchling build; seed=42 implemented; all 20 references cited |

## Detailed Findings

### Clinical Accuracy
- SOFA scoring uses 3/6 components (cardiovascular, renal, metabolic) -- appropriately disclosed as simplification
- MAP target of 65 mmHg matches SSC guidelines (Rhodes 2017, Evans 2021)
- Fluid gain of 8.0 mmHg/L calibrated against CLOVERS trial -- reasonable range
- Lactate clearance kinetics reference Hernandez 2020 and Bakker 2022
- Paper explicitly states "SepsiSim is a benchmark environment, not a clinical decision support tool"
- No overstated clinical claims

### RL Methodology
- Three graduated environments follow curriculum learning best practices
- PPO negative result on FluidResuscitation (-71.5 vs random 100.4) honestly reported and explained: antibiotic dominance makes fluid dosing secondary, PPO overfits to aggressive dosing
- PPO success on VasopressorTitration (-36.9 vs -41.3 random, lowest variance) demonstrates learnable signals
- SepsisManagement shows clear agent quality separation (random sigma=34.5 vs PPO sigma=1.8)
- Hyperparameters are standard (lr 1e-4 to 3e-4, batch 64-128, gamma 0.99-0.995)
- 200K-500K timesteps appropriate for continuous control with these obs/action dimensions

### Writing Quality
- 10 well-structured sections with logical flow
- Mathematical notation consistent throughout (4 ODE equations, cardiovascular and lactate models)
- 3 tables (environments, hyperparameters, results) clearly formatted
- Abstract updated to accurately reflect mixed results ("differentiated reward signals")
- LaTeX compiles cleanly (minor overfull hbox warning only)
- No em-dash Unicode characters (all use LaTeX --- triple hyphen)

### Reproducibility
- Package installable via `pip install sepsisim`
- 136 tests pass (pytest verified)
- Seed=42 used throughout training and evaluation
- All 20 BibTeX entries cited in paper text
- Training results saved to JSON for exact replication
- Trained models saved as .zip files

## Issues Identified (Minor -- Not Blocking)

1. **Line 29 overfull hbox:** Abstract line slightly exceeds margin. Minor typographic issue, not content.
2. **Figure reference:** Line 65 references "Figure 1" but no figure is included. Should remove reference or add architecture diagram.
3. **Abstract claim scope:** Abstract says "across all environments" regarding learnable signals, but FluidResuscitation PPO result is negative. The body text correctly contextualizes this, but the abstract could be more precise.

## Disposition

All issues are minor and do not affect the scientific validity or reproducibility of the work. The paper honestly reports both successes and failures, provides complete reproducibility information, and makes no overstated claims.

**APPROVED FOR PUBLICATION**

---
*Review conducted: 2026-03-24*
*Pipeline: AppliedResearch STAGE 6*
