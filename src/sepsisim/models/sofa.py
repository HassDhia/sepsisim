"""SOFA (Sequential Organ Failure Assessment) score computation.

Implements a simplified SOFA score based on available simulation variables.

SIMPLIFICATION: Using 3 of 6 SOFA components (cardiovascular, hepatic via
lactate proxy, renal via urine output). Full clinical SOFA includes
respiratory (PaO2/FiO2), coagulation (platelets), and hepatic (bilirubin).
Acceptable for benchmark simulation focused on hemodynamic management.

Reference:
    Singer, M., Deutschman, C.S., Seymour, C.W., et al. (2016).
    The Third International Consensus Definitions for Sepsis and Septic Shock
    (Sepsis-3). JAMA, 315(8), 801-810.
"""

from __future__ import annotations


def cardiovascular_sofa(map_mmhg: float, vasopressor_dose: float) -> int:
    """Cardiovascular SOFA component (0-4).

    Args:
        map_mmhg: mean arterial pressure in mmHg.
        vasopressor_dose: norepinephrine-equivalent dose (mcg/kg/min).

    Returns:
        SOFA score component (0-4).
    """
    if vasopressor_dose > 0.3:
        return 4
    elif vasopressor_dose > 0.1:
        return 3
    elif vasopressor_dose > 0.0:
        return 2
    elif map_mmhg < 70:
        return 1
    return 0


def renal_sofa(urine_output_ml_kg_h: float) -> int:
    """Renal SOFA component based on urine output (0-4).

    Args:
        urine_output_ml_kg_h: urine output in mL/kg/h.

    Returns:
        SOFA score component (0-4).
    """
    if urine_output_ml_kg_h < 0.1:
        return 4
    elif urine_output_ml_kg_h < 0.2:
        return 3
    elif urine_output_ml_kg_h < 0.3:
        return 2
    elif urine_output_ml_kg_h < 0.5:
        return 1
    return 0


def lactate_sofa(lactate_mmol_l: float) -> int:
    """Hepatic/metabolic SOFA proxy based on lactate (0-4).

    Note: Clinical SOFA uses bilirubin for hepatic scoring.
    We use lactate as a proxy for tissue perfusion adequacy,
    which correlates with overall organ dysfunction in sepsis.

    Args:
        lactate_mmol_l: blood lactate in mmol/L.

    Returns:
        SOFA-like score component (0-4).
    """
    if lactate_mmol_l > 10.0:
        return 4
    elif lactate_mmol_l > 6.0:
        return 3
    elif lactate_mmol_l > 4.0:
        return 2
    elif lactate_mmol_l > 2.0:
        return 1
    return 0


def compute_sofa_score(
    map_mmhg: float,
    vasopressor_dose: float,
    urine_output_ml_kg_h: float,
    lactate_mmol_l: float,
) -> int:
    """Compute simplified SOFA score (0-12).

    Uses 3 components: cardiovascular, renal, and lactate-based metabolic.

    Args:
        map_mmhg: mean arterial pressure (mmHg).
        vasopressor_dose: norepinephrine-equivalent dose (mcg/kg/min).
        urine_output_ml_kg_h: urine output (mL/kg/h).
        lactate_mmol_l: blood lactate (mmol/L).

    Returns:
        Simplified SOFA score (0-12, sum of 3 components each 0-4).
    """
    return (
        cardiovascular_sofa(map_mmhg, vasopressor_dose)
        + renal_sofa(urine_output_ml_kg_h)
        + lactate_sofa(lactate_mmol_l)
    )
