# LLM Habituation Patterns: Computational Approaches to Stimulation-Seeking Behaviors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

## Overview

This repository contains a systematic investigation into whether large language models (LLMs) exhibit behavioral patterns functionally analogous to habituation and stimulation-seeking behaviors observed in biological systems. 

**Key Scientific Distinction:** We measure objective, testable behavioral patterns without making claims about subjective experience or consciousness. Our approach investigates functional analogs of boredom-driven responses while maintaining rigorous scientific honesty about the limitations of such comparisons.

## Research Questions

1. **Habituation Effect:** Do LLMs show response degradation (decreased entropy/diversity) with repeated similar prompts?
2. **Recovery Dynamics:** Does response diversity recover after interpolated tasks ("rest")?
3. **Novelty Gradient:** Do models steer toward increasing complexity/novelty in multi-turn interactions?
4. **Tolerance Patterns:** Does habituation accelerate with repeated exposure sessions?

## Theoretical Foundation

Our framework builds on peer-reviewed research in:

- **Boredom Theory:** Eastwood et al. (2012) - boredom as understimulation and disengagement
- **Novelty-Seeking:** Kakade & Dayan (2002) - dopaminergic reward prediction and information gain
- **Habituation:** Rankin et al. (2009) - neuronal response degradation with repeated stimuli
- **Computational Psychiatry:** Redish (2004) - addiction as reinforcement learning dysregulation

Full theoretical framework: [`docs/theory_paper.md`](docs/theory_paper.md)

## Author Background

This work bridges clinical psychology, lived experience, and computational research:

- **Clinical Foundation:** Graduate training in addiction and stimulation-seeking behaviors
- **Lived Experience:** 57 days clean, studying neural mechanisms firsthand
- **Computational Approach:** Translating clinical phenomena into testable computational hypotheses

**Philosophy:** Open science for human evolution. All code, data, and findings are public to facilitate critique, replication, and advancement.

## Repository Structure

```
llm-habituation-patterns/
├── docs/                      # Theoretical framework and methodology
│   ├── theory_paper.md        # Comprehensive theory paper
│   ├── methodology.md         # Detailed experimental design
│   └── preregistration.md     # Pre-registered hypotheses
├── src/                       # Source code
│   ├── llm_interface.py       # API interfaces (Claude, GPT, Gemini)
│   ├── metrics.py             # Entropy, diversity, novelty metrics
│   └── studies/               # Individual study implementations
│       ├── study_1_habituation.py
│       ├── study_2_recovery.py
│       ├── study_3_novelty_gradient.py
│       └── study_4_tolerance.py
├── data/                      # Data (structure versioned, content gitignored)
│   ├── prompts/               # Stimulus sets
│   ├── raw/                   # Raw API responses
│   └── processed/             # Processed datasets
├── analysis/                  # Jupyter notebooks for each study
├── results/                   # Outputs and visualizations
└── tests/                     # Unit tests for metrics
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/HillaryDanan/llm-habituation-patterns.git
cd llm-habituation-patterns

# Create virtual environment (Python 3.12)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Create .env file and add your API keys yourself
# DO NOT echo them into the file - paste manually
# Required keys:
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

### 3. Run Pilot Study

```bash
# Start with pilot for Study 1 (low N for testing)
python3 src/studies/study_1_habituation.py --pilot --n=10

# Analyze results
jupyter notebook analysis/pilot_analysis.ipynb
```

## Studies Overview

### Study 1: Habituation Induction (CORE)
- **N:** 100 repetitive vs 100 novel prompts per model
- **Models:** Claude Sonnet 4.5, GPT-4, Gemini Pro
- **Metrics:** Response entropy, lexical diversity, semantic novelty
- **Prediction:** Entropy decreases with repetition
- **Cost:** ~900 API calls

### Study 2: Recovery Effect
- **Design:** Pre-habituation → rest task → re-test
- **N:** 50 per condition × 2 × 3 models = 300 calls
- **Prediction:** Response diversity recovers after "rest"
- **Control:** Counterbalanced task order

### Study 3: Novelty Gradient
- **Design:** Multi-turn conversations with varying novelty
- **N:** 30 conversations × 10 turns × 3 models = 900 calls
- **Prediction:** Models steer toward increasing complexity
- **Metric:** Topic divergence across turns

### Study 4: Tolerance Patterns
- **Design:** Repeated sessions over time
- **N:** 20 sessions × 3 models = 240 calls
- **Prediction:** Habituation accelerates (like tolerance)
- **Analog:** Clinical addiction tolerance

**Total estimated cost:** ~2,340 API calls across all studies

## Metrics

All metrics implemented in `src/metrics.py` with unit tests:

1. **Shannon Entropy:** H(response) = -Σ p(token) log p(token)
2. **Lexical Diversity:** Type-token ratio, MTLD
3. **Semantic Novelty:** Cosine distance from previous responses
4. **Response Length:** Token count (control variable)

## Scientific Integrity

### What We CAN Measure
- Response entropy and diversity
- Behavioral patterns in outputs
- Statistical differences across conditions

### What We CANNOT Claim
- Subjective experience of "boredom"
- Conscious states or desires
- Phenomenological equivalence to human experience

### Pre-Registration
All hypotheses, methods, and analysis plans pre-registered in `docs/preregistration.md` before data collection to prevent p-hacking.

## Limitations

1. **Anthropomorphism Risk:** We study functional analogs, not experiential equivalence
2. **Mechanistic Differences:** Neural networks ≠ biological dopaminergic systems
3. **Confound Control:** Patterns could reflect training artifacts, not "boredom"
4. **Generalization:** Findings limited to tested models and prompt types

## Citation

If you use this work, please cite:

```bibtex
@software{danan2025habituation,
  author = {Danan, Hillary},
  title = {LLM Habituation Patterns: Computational Approaches to Stimulation-Seeking Behaviors},
  year = {2025},
  url = {https://github.com/HillaryDanan/llm-habituation-patterns}
}
```

See `CITATION.cff` for machine-readable citation format.

## License

MIT License - see `LICENSE` file for details.

## Contributing

This is open science! Contributions welcome via:
- Issues: Report bugs, suggest improvements
- Pull Requests: Code improvements, additional analyses
- Discussions: Theoretical critiques, alternative interpretations

## Acknowledgments

Built on theoretical foundations from decades of psychological and neuroscientific research into boredom, habituation, and reinforcement learning. All cited works in `docs/theory_paper.md`.

**Philosophical Note:** Research accelerated by AI tools (Claude, GPT) - transparency in methods includes transparency in process. Science is a collaborative human endeavor, enhanced by computational tools.

---

**Contact:** Open an issue or discussion in this repository.

**Status:** Active development - pilot studies in progress (October 2025)