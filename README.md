# Response Invariance in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

## Overview

This repository contains a systematic investigation into temporal response dynamics in large language models (LLMs). Through four pre-registered studies (N=330 prompts across Claude Sonnet 4.5 and GPT-4), we tested whether LLMs exhibit habituation, sensitization, recovery, and tolerance—temporal dynamics ubiquitous in biological learning systems.

**Key Finding:** LLMs exhibit **response invariance**—statistical stability of output distributions across temporal manipulations—revealing fundamental architectural differences from biological neural networks.

## Main Results

### Comprehensive Null Findings

- **No habituation** (Study 1: β≈0, p=0.91, d<0.1, N=100)
- **No recovery effect** (Study 2: 1.2% change, N=70)
- **No sensitization** (Pilot: p=0.26, N=15)
- **No tolerance** (Study 4: flat across sessions, N=60)

Meta-analysis: Mean |d|=0.06 across all studies, despite >0.85 statistical power for detecting small effects.

### One Positive Finding

**Conversational convergence** (Study 3): Semantic novelty decreased 36% over multi-turn interactions (p<0.001), consistent with conversational grounding rather than exploration or novelty-seeking.

## Significance

Response invariance has implications for:

- **AI Architecture:** Establishes fundamental property of transformer-based systems
- **AI Safety:** Models lack adaptive mechanisms present in biological systems
- **Human-AI Interaction:** Informs design of personalization and memory systems
- **Computational Neuroscience:** Constrains which cognitive phenomena LLMs can model

## Repository Structure

```
llm-habituation-patterns/
├── docs/
│   ├── papers/                           # Manuscript and supplementary materials
│   │   ├── response_invariance_manuscript.md
│   │   ├── supplementary_materials.md
│   │   └── cover_letter.md
│   ├── framework_response_invariance.md  # Theoretical framework
│   ├── followup_studies_plan.md          # Future research directions
│   └── DESIGN_CHANGES.md                 # Design evolution documentation
├── src/
│   ├── llm_interface.py                  # Unified API wrapper
│   ├── metrics.py                        # Entropy, diversity, novelty metrics
│   └── studies/                          # Study implementations
│       ├── study_1_habituation.py
│       ├── study_2_recovery.py
│       ├── study_3_novelty_gradient.py
│       ├── study_4_tolerance.py
│       └── study_sensitization.py
├── analysis/
│   └── generate_figures.py               # Publication-quality figures
├── data/                                 # Data files (see .gitignore)
└── results/                              # Outputs and visualizations
```

## Quick Start

### Installation

```bash
git clone https://github.com/HillaryDanan/llm-habituation-patterns.git
cd llm-habituation-patterns

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create `.env` file with API keys:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### Run Studies

```bash
# Study 1: Test for habituation (within-subjects design)
python3 src/studies/study_1_habituation.py --models claude gpt4

# Study 2: Test for recovery effects
python3 src/studies/study_2_recovery.py --models claude

# Study 3: Multi-turn conversation dynamics
python3 src/studies/study_3_novelty_gradient.py --models claude

# Study 4: Session-level tolerance patterns
python3 src/studies/study_4_tolerance.py --models claude
```

### Generate Figures

```bash
python3 analysis/generate_figures.py
```

## Study Designs

### Study 1: Habituation Test (N=100)

**Design:** Within-subjects, 10 concepts × 10 identical repetitions  
**Hypothesis:** Response entropy declines with repetition  
**Result:** No systematic decline (β=0.000071, p=0.91)  
**Conclusion:** No evidence for habituation

### Study 2: Recovery Effect (N=70)

**Design:** Habituation phase → rest task → re-test  
**Hypothesis:** Response diversity recovers after rest  
**Result:** Minimal change (1.2%, p=0.18)  
**Conclusion:** No recovery effect detected

### Study 3: Multi-Turn Dynamics (N=300)

**Design:** 30 conversations × 10 turns each  
**Hypothesis:** Increasing semantic novelty (exploration)  
**Result:** Decreasing novelty (36% reduction, p<0.001)  
**Conclusion:** Convergence pattern (grounding, not exploration)

### Study 4: Tolerance Patterns (N=60)

**Design:** 3 sessions × 20 prompts each  
**Hypothesis:** Accelerating response decrement  
**Result:** Flat across sessions (F=0.82, p=0.45)  
**Conclusion:** No tolerance development

## Methods

**Response Diversity:** Shannon entropy (normalized)  
**Lexical Diversity:** MTLD, Type-Token Ratio  
**Semantic Novelty:** Cosine distance in embedding space  
**Statistical Analysis:** Linear regression, mixed-effects models, equivalence testing

All methods pre-registered. Design refinements fully documented in `docs/DESIGN_CHANGES.md`.

## Theoretical Framework

**Response Invariance:** Statistical stability of LLM outputs across temporal manipulations, reflecting stateless processing architecture.

**Architectural Basis:** Transformers process inputs independently without persistent state changes. Each generation samples from frozen parameter distributions, precluding temporal dynamics.

**Conversational Grounding:** The single temporal pattern (Study 3 convergence) reflects self-attention over conversation history—context-dependent processing, not temporal adaptation.

See `docs/framework_response_invariance.md` for complete framework.

## Manuscript Status

**In Preparation:** Manuscript and supplementary materials available in `docs/papers/`

**Pre-print:** Coming soon (arXiv/PsyArxiv)

## Scientific Integrity

### Pre-Registration

All hypotheses, methods, and analyses specified before data collection. Available in `docs/preregistration.md`.

### Methodological Transparency

- Design evolution documented (`docs/DESIGN_CHANGES.md`)
- Pilot findings informed refinements
- All decisions justified with peer-reviewed literature

### Power Analysis

Studies designed with >0.85 power for small-to-medium effects. Observed effects an order of magnitude smaller, confirming true null findings.

### Open Science

- All code publicly available
- All data available (subject to API terms of service)
- Reproducible analysis pipeline
- Complete transparency in methods

## Future Directions

Six follow-up studies designed to test positive predictions from response invariance framework:

- Context window position effects
- Conversational grounding mechanisms
- Prompt structure manipulation
- Temperature as exploration parameter
- External memory systems
- Alternative architectures

See `docs/followup_studies_plan.md` for details.

## Citation

```bibtex
@software{danan2025responseinvariance,
  author = {Danan, Hillary},
  title = {Response Invariance in Large Language Models: 
           Comprehensive Evidence from Four Pre-Registered Studies},
  year = {2025},
  url = {https://github.com/HillaryDanan/llm-habituation-patterns},
  note = {Manuscript in preparation}
}
```

See `CITATION.cff` for machine-readable format.

## License

MIT License - see `LICENSE` for details.

## Contributing

Contributions welcome via:

- **Issues:** Bug reports, methodological questions
- **Pull Requests:** Code improvements, additional analyses
- **Discussions:** Theoretical interpretations, replication attempts

## References

Key papers informing this work:

- Rankin et al. (2009). Habituation revisited. *Neurobiol. Learn. Mem.*
- Vaswani et al. (2017). Attention is all you need. *NeurIPS*
- Clark & Brennan (1991). Grounding in communication. In *Perspectives on Socially Shared Cognition*
- Lakens (2017). Equivalence tests. *Soc. Psychol. Personal. Sci.*

Complete references in `docs/papers/response_invariance_manuscript.md`

## Contact

Open an issue or start a discussion in this repository.

---

**Repository Status:** Active (October 2025)  
**Manuscript Status:** In preparation for journal submission
