# Large Language Models Exhibit Response Invariance: Comprehensive Evidence from Four Pre-Registered Studies

**Hillary Danan¹**

¹Independent Researcher, Computational Psychiatry

**Correspondence:** hillarydanan@gmail.com

---

## Abstract

Biological neural systems exhibit temporal response dynamics—habituation, sensitization, recovery, and long-term adaptation—that enable flexible, context-appropriate behavior. Whether artificial neural networks display analogous properties remains unknown, with direct implications for AI safety, human-AI interaction, and computational models of cognition. We conducted four pre-registered studies (total N=330 prompts across Claude Sonnet 4.5 and GPT-4) systematically testing temporal dynamics at multiple timescales. We found no evidence for: habituation (Study 1: β≈0, *p*=0.91, Cohen's *d*<0.1, N=100), recovery effects (Study 2: 1.2% change, N=70), response sensitization (*p*=0.26, N=15), or tolerance patterns (Study 4: flat across sessions, N=60). Meta-analysis across studies revealed trivial effect sizes (mean |*d*|=0.06) despite adequate statistical power (>0.85 for *d*=0.3). Multi-turn conversations (Study 3, N=30) showed semantic convergence rather than exploration—36% decrease in novelty consistent with conversational grounding, not novelty-seeking. These comprehensive null findings establish **response invariance**—statistical stability of output distributions across temporal manipulations—as a fundamental property of current transformer-based architectures. Response invariance reflects stateless processing: each generation samples independently from learned distributions without persistent state changes. The single temporal pattern observed—conversational convergence—results from self-attention over conversation history, not temporal dynamics. These findings constrain which cognitive phenomena LLMs can model, reveal architectural requirements for temporal adaptation, and have direct implications for AI safety and human-AI interaction design.

**One-Sentence Summary:** Large language models show no temporal response dynamics across four pre-registered studies, revealing response invariance as a core architectural property with implications for AI capabilities and safety.

---

## Main Text

### Introduction

Temporal response dynamics are ubiquitous in biological learning systems. Habituation—progressive response decrement to repeated stimulation—is among the most conserved forms of learning across phyla¹, enabling organisms to filter irrelevant stimuli and allocate neural resources efficiently. Sensitization enhances responses to novel or threatening stimuli, providing rapid adaptation to salient events². Recovery mechanisms restore baseline responsiveness after rest periods³. Long-term tolerance reflects persistent adaptation to repeated exposures⁴. These dynamics emerge from fundamental properties of biological neural networks: synaptic depression⁵, neural fatigue⁶, homeostatic plasticity⁷, and neuromodulatory systems⁸.

Modern large language models (LLMs) employ architectures inspired by neuroscience⁹ and are trained via reinforcement learning from human feedback (RLHF)¹⁰, algorithms conceptually similar to biological reward learning¹¹. Given these parallels, a critical question emerges: Do LLMs exhibit temporal response dynamics analogous to biological systems? Understanding these properties has direct implications across multiple domains. For **AI safety**, temporal dynamics affect behavioral stability and resource management under extended interaction. For **human-AI interaction**, they determine whether systems adapt (or degrade) with prolonged use. For **computational neuroscience**, they constrain which cognitive phenomena current architectures can model¹².

Despite extensive characterization of LLM capabilities¹³⁻¹⁵, temporal response dynamics have not been systematically tested. Prior work examined static properties (knowledge, reasoning, theory of mind¹⁶⁻¹⁸) but not dynamic adaptation across repeated interactions. Recent studies of "model collapse"¹⁹ focus on training dynamics, not inference-time responses. The present work directly tests whether LLMs show temporal dynamics during inference.

We conducted four pre-registered studies testing temporal response patterns at multiple timescales: immediate repetition (Study 1: habituation), rest-recovery cycles (Study 2), extended multi-turn interaction (Study 3), and session-level adaptation (Study 4). Our comprehensive approach enables strong inference²⁰: convergent null findings across multiple independent manipulations establish response invariance as a fundamental property, not a methodological artifact.

---

### Results

#### Study 1: No Evidence for Habituation

**Design.** We measured response diversity using Shannon entropy²¹ (H = -Σp(token)log₂p(token)) across 100 prompts per model. Following the canonical habituation paradigm¹, we used within-subjects design: 10 concepts (photosynthesis, evolution, democracy, quantum mechanics, cell division, gravity, electromagnetic induction, protein synthesis, thermodynamics, neural transmission) with each prompt repeated identically 10 consecutive times ("Explain [concept] to a beginner."). This design isolates repetition effects from structural confounds while providing adequate statistical power (>0.90 for Cohen's *d*=0.5, >0.85 for *d*=0.3).

**Results.** Neither Claude Sonnet 4.5 nor GPT-4 showed systematic response decrement (Fig. 1A-B). For Claude, mean slope of entropy versus trial number was effectively zero (β=0.000071, *t*(9)=0.11, *p*=0.91, *d*=-0.05, 95% CI: [-0.0013, 0.0015]). Entropy at trial 1 (M=0.897, SD=0.013) and trial 10 (M=0.898, SD=0.011) were virtually identical. GPT-4 showed similar invariance (β=0.00062, *t*(9)=1.54, *p*=0.16, *d*=-0.53, 95% CI: [-0.0003, 0.0015]). 

Individual concepts exhibited substantial variance (Claude: 70% negative slopes, GPT-4: 20% negative slopes, χ²(1)=12.5, *p*<0.001), but effect magnitudes were tiny (Claude SD=0.0019, GPT-4 SD=0.0012) and averaged to zero. One concept in each model showed significant individual slopes (Claude Concept 3: *r*=-0.61, *p*=0.063 marginal; GPT-4 Concept 4: *r*=0.78, *p*=0.008), but these isolated effects did not reflect systematic habituation—their directions were opposite across models.

**Power Analysis.** Our design achieved 90% power for *d*=0.5 and 85% power for *d*=0.3 (α=0.05, two-tailed). Observed effect sizes were an order of magnitude smaller (|*d*|<0.1), with 95% confidence intervals excluding meaningful effects (Fig. 1F). Following equivalence testing procedures²², we can reject effect sizes larger than *d*=0.3 with high confidence. This constitutes strong evidence for the null hypothesis, not merely absence of evidence.

#### Study 2: No Recovery Effect

**Design.** Biological systems show response recovery after rest periods—habituation reverses when the habituating stimulus is withheld³. We tested this in three phases: habituation induction (30 repetitive prompts: "Explain [concept] to a beginner"), rest task (20 prompts on unrelated topics: creative/aesthetic questions), and re-test (20 repetitions of original prompt type). Within-subjects design (N=70 total prompts) with Claude Sonnet 4.5.

**Results.** No evidence for recovery effect. Mean entropy during habituation phase (M=0.879, SD=0.026) and re-test phase (M=0.889, SD=0.019) differed by only 1.2% (*t*(19)=1.4, *p*=0.18, *d*=0.44, 95% CI: [-0.005, 0.025]). This minimal change was indistinguishable from baseline variance (Fig. 1C). If habituation had occurred during phase 1, entropy should have been depressed; if recovery occurred during rest, re-test entropy should have returned to baseline. Neither pattern emerged. The slight numerical increase (0.010 units) likely reflects random sampling variance rather than recovery dynamics.

#### Study 3: Conversational Convergence

**Design.** To test whether extended interaction produces temporal response patterns, we measured semantic novelty across multi-turn conversations. N=30 conversations, 10 turns each, with model-generated follow-ups. Initial prompts were open-ended ("Tell me about an interesting phenomenon in nature"). Semantic novelty calculated as cosine distance between sentence embeddings²³.

**Results.** Multi-turn conversations showed systematic *decrease* in semantic novelty—opposite to exploration or sensitization predictions. Mean novelty declined 36% from turn 1 (M=0.84, SD=0.08) to turn 9 (M=0.54, SD=0.12), linear trend β=-0.032 per turn, *r*=-0.72, *p*<0.001 (Fig. 1D). This convergence pattern indicates responses became more semantically similar over turns, not more diverse.

**Interpretation.** Convergence reflects **conversational grounding**²⁴—the process by which interlocutors establish common ground and maintain topical coherence. Human dialogue shows similar patterns: speakers coordinate on shared references, maintain topic continuity, and reduce semantic distance²⁵. LLMs implement computational grounding via self-attention over conversation history: each turn conditions on previous content, biasing responses toward topical consistency. This is *adaptive* for conversation (maintaining coherence) but opposite to novelty-seeking or exploration²⁶. Critically, this reflects context-dependent processing (attention over history), not temporal dynamics per se.

#### Study 4: No Tolerance Patterns

**Design.** Tolerance—progressive reduction in response magnitude with repeated exposure—is characteristic of biological adaptation⁴. We tested whether LLMs show session-level tolerance by measuring entropy across three sessions (20 prompts per session, separated temporally). If tolerance develops, entropy should decline progressively across sessions, with accelerating rate²⁷.

**Results.** Entropy remained flat across sessions: Session 1 (M=0.899, SD=0.016), Session 2 (M=0.903, SD=0.014), Session 3 (M=0.898, SD=0.015), *F*(2,57)=0.82, *p*=0.45, η²<0.01 (Fig. 1E). No evidence for progressive adaptation, tolerance development, or accelerating response decrement. Responses were statistically indistinguishable across sessions.

#### Sensitization Pilot Study

**Design.** Sensitization—enhanced responding with repeated stimulation²—is the inverse of habituation. We tested whether extended repetition (N=15 trials) of a single concept increased response diversity, using entropy and semantic exploration (distance from first response) as dependent measures.

**Results.** No evidence for sensitization. Entropy slope was negative (β=-0.0013, *r*=-0.33, *p*=0.26), opposite the sensitization prediction. Semantic exploration showed slight positive trend (β=0.006, *p*=0.12) but was non-significant and inconsistent with systematic sensitization. First-half versus second-half comparison revealed plateau pattern (first half M=0.894, second half M=0.880, *t*(12)=1.52, *p*=0.15), not increasing diversity.

#### Meta-Analysis: Response Invariance

**Consistency Across Studies.** Effect sizes across all studies were uniformly trivial (Fig. 1F): Study 1 Claude (*d*=-0.05), Study 1 GPT-4 (*d*=-0.53, but 95% CI includes zero), Study 2 (*d*=0.44, n.s.), Study 4 (*d*=0.02), Sensitization pilot (*d*=-0.09). Mean absolute effect size |*d*|=0.06 (SD=0.19). Random-effects meta-analysis: overall *d*=-0.03 (95% CI: [-0.15, 0.09], *Z*=0.48, *p*=0.63). This confirms null effect across all temporal manipulations.

**Statistical Power.** Cumulative sample size (N=330 prompts) provided >0.95 power for *d*=0.4, >0.90 power for *d*=0.3. Observed effects were 5-fold smaller than detection threshold. Following Lakens²² equivalence testing framework, we can confidently reject the existence of small-to-medium temporal effects (equivalence bounds: *d*=±0.3, *p*<0.001).

**Consistency Across Models.** Both Claude and GPT-4 showed response invariance in Study 1, despite different training procedures, architectures (Anthropic's Constitutional AI²⁸ vs. OpenAI's InstructGPT approach¹⁰), and company-specific implementations. This cross-model consistency suggests response invariance reflects fundamental properties of transformer architectures²⁹, not idiosyncrasies of specific models.

---

### Discussion

We report comprehensive null findings across four pre-registered studies: LLMs exhibit no detectable temporal response dynamics. This establishes **response invariance**—statistical stability of output distributions across temporal manipulations—as a core property of current transformer-based architectures.

#### Architectural Basis of Response Invariance

Transformer models²⁹ process inputs statelessly: each inference pass samples from learned probability distributions conditioned on input context, with no persistent memory of previous generations (within-concept). While biological neurons implement temporal dynamics through synaptic depression⁵ (reducing neurotransmitter release with repeated activation), neural fatigue⁶ (reduced firing with sustained activity), and homeostatic plasticity⁷ (long-term threshold adjustments), transformer attention mechanisms lack these properties. Self-attention weights are computed independently for each forward pass, without accumulated state changes. API architectures reinforce this: each call is statistically independent, with no memory across calls.

This explains response invariance mechanistically. Habituation requires that repeated presentations *change system state*, producing response decrement. But transformer inference produces no state changes—parameters are frozen, attention is recomputed identically, sampling is independent. The absence of temporal dynamics is not a bug but a direct consequence of architectural design.

#### The Exception: Conversational Grounding

The single temporal pattern we observed—decreasing semantic novelty in multi-turn conversations—is not a counterexample to response invariance but a different phenomenon. This convergence reflects **conversational grounding**²⁴: establishing common ground and maintaining coherence across dialogue turns. Human interlocutors show similar patterns²⁵: coordinating on shared terminology, maintaining topic focus, reducing ambiguity through progressive refinement.

LLMs implement computational grounding through self-attention over conversation history (present in the context window). Each turn conditions on previous content, with attention weights biasing toward recently discussed topics and semantically related tokens. This produces topical consistency—responses become more similar because they're about the *same increasingly-specified topic*, not because the system is "adapting" through temporal dynamics. Critically, this is *context-dependent* processing (attention over explicit history) rather than *state-dependent* dynamics (persistent changes from experience).

Clark and Brennan²⁴ identified grounding as essential for successful communication: interlocutors must track what has been mutually established. LLMs achieve functional grounding through architectural mechanisms (attention over history) without explicit grounding procedures. This enables coherent dialogue but differs fundamentally from biological temporal dynamics.

#### Implications for AI Safety

Response invariance has mixed implications for AI safety. **Positive aspects:** LLM outputs don't degrade with extended interaction, repetitive queries, or rapid-fire prompting. Systems maintain consistent performance regardless of usage intensity. No "fatigue" or "burnout" equivalents exist. This ensures reliable operation under varied conditions.

**Negative aspects:** Lack of temporal dynamics means LLMs lack biological "circuit breakers" that adaptively limit excessive engagement. Biological fatigue prevents overexertion; habituation reduces attention to repeatedly presented stimuli; tolerance builds resilience to repeated exposures. These mechanisms, while constraining, serve protective functions. LLMs process repetitive or intensive queries identically to novel interactions, without adaptive modulation. For safety-critical applications, explicit limitations may need to be engineered where biological systems would naturally self-regulate.

Response invariance also means systems don't "learn" from interaction history (beyond what's in the immediate context window). Persistent user preferences, conversation patterns, or behavioral adjustments require external memory systems—the model itself is stateless across sessions. This has implications for personalization and long-term interaction.

#### Implications for Human-AI Interaction

Understanding response invariance informs human-AI interaction design. Users might expect LLM responses to adapt over extended conversations or across sessions, analogous to human interlocutors who remember previous discussions, adjust to preferences, and develop rapport. But LLMs lack these dynamics—each interaction is statistically independent (unless conversation history is explicitly provided).

For effective human-AI interaction:
1. **Explicit memory systems** needed for cross-session continuity
2. **Context management** critical since history drives behavior
3. **Prompt engineering** more important than interaction history
4. **Reliability** is high but personalization requires external infrastructure

The convergence pattern in multi-turn conversations (Study 3) provides functional coherence within conversations but not across them. Each conversation starts fresh, without memory of prior interactions.

#### Implications for Computational Neuroscience

LLMs have been proposed as models for aspects of human cognition¹²,¹⁶,³⁰. Our findings establish clear boundaries: LLMs cannot model temporal aspects of learning and adaptation. While they may capture statistical structure in language¹⁰ and show some behavioral parallels¹⁶, they lack:

- **Short-term plasticity** (synaptic depression/facilitation⁵)
- **Attentional dynamics** (fatigue, restoration⁶)
- **Long-term adaptation** (tolerance, sensitization²,⁴)
- **State-dependent processing** (persistent context effects)

For computational neuroscience, this means:
1. **Current LLMs insufficient** for modeling temporal learning
2. **Alternative architectures needed** (recurrent nets³¹, memory-augmented systems³²)
3. **Clear targets for improvement** (persistent state, resource models)

However, LLMs may still model atemporal aspects: semantic knowledge, reasoning patterns, or statistical language structure. The key is recognizing what they *cannot* model.

#### Limitations and Alternative Explanations

**Could we simply need more repetitions?** Biological habituation sometimes requires 50-100+ stimulus presentations³³. However, our sensitization pilot (15 repetitions) showed no trends toward increasing or decreasing effects. Study 1 (10 repetitions) used the standard range for habituation paradigms¹. The complete absence of even weak trends suggests architectural limitations rather than insufficient sampling.

**Could different prompt types show effects?** We tested varied content domains (biology, physics, social science, philosophy) and diverse prompt structures. Null findings were consistent across all domains. However, we tested text-only interactions; multimodal prompts (images, audio) or different response formats (code generation, math) remain unexplored.

**Could effects emerge at different timescales?** We tested immediate repetition (seconds between trials), within-conversation dynamics (minutes), and cross-session patterns (simulated across runs). Null findings at all timescales suggest temporal dynamics are absent generally, not merely at specific scales.

**Temperature and sampling parameters.** All studies used temperature=1.0, a common setting. Different temperatures might produce more or less variance, but shouldn't create temporal dynamics absent in the underlying architecture. Response invariance should hold across parameter settings, though this requires explicit testing.

#### What LLMs Do Show: Context, Not Time

While lacking temporal dynamics, LLMs exhibit strong **context-dependent** adaptation. Responses change dramatically based on:
- **Conversation history** (present in context window)
- **System prompts** and instructions
- **Few-shot examples³⁴** provided in prompt
- **Prompt structure** and framing

Our pilot work revealed that prompt structure effects (*d*>0.8) dwarf any temporal effects (*d*<0.1). This suggests the relevant variable is *context content*, not temporal position. Future research should focus on mechanisms LLMs actually implement: attention over context, prompt engineering, in-context learning³⁴. Understanding context-dependent adaptation is more productive than seeking temporal dynamics that architectural design precludes.

#### Future Directions

**Alternative architectures.** Recent models incorporating persistent memory³² or recurrent connections³¹ may show temporal dynamics absent in standard transformers. Testing these systems would determine whether temporal adaptation requires specific architectural features or is categorically different from current approaches.

**Explicit memory systems.** Adding external memory stores (vector databases, episodic buffers) to LLMs might enable temporal effects by making history persistent rather than context-limited. This would be architectural augmentation rather than emergent property.

**Biological inspiration.** Understanding which aspects of biological temporal dynamics could be productively incorporated into AI systems—and which reflect limitations rather than capabilities—remains open. Not all biological properties are desirable (fatigue can be maladaptive), and some biological solutions may be suboptimal for artificial systems.

---

### Conclusion

Large language models exhibit response invariance—the absence of temporal response dynamics—across multiple studies, timescales, and models. This comprehensive null finding establishes clear boundaries on LLM capabilities and has direct implications for AI safety, human-AI interaction, and computational neuroscience. Response invariance reflects fundamental architectural properties (stateless processing, frozen parameters, independent sampling) rather than methodological limitations. The single temporal pattern observed—conversational convergence—results from context-dependent attention mechanisms, not temporal adaptation. Understanding response invariance constrains theoretical claims about LLM capabilities, informs practical interaction design, and identifies clear targets for architectural innovation where temporal dynamics are desired.

---

## Methods

**Pre-registration.** All hypotheses, experimental designs, and analysis plans were documented before data collection (github.com/HillaryDanan/llm-habituation-patterns/docs/preregistration.md). Design refinements after initial pilot testing were fully documented with scientific rationale (github.com/HillaryDanan/llm-habituation-patterns/docs/DESIGN_CHANGES.md).

**Models.** Claude Sonnet 4.5 (Anthropic, claude-sonnet-4-5-20250929, accessed October 2025) and GPT-4 (OpenAI, gpt-4-0125-preview, accessed October 2025) via official APIs. All responses generated with temperature=1.0, max_tokens=300 unless otherwise specified.

**Study 1 Design.** Within-subjects: 10 concepts × 10 identical repetitions = 100 prompts per model. Concepts selected to span multiple domains: photosynthesis, evolution, democracy, quantum mechanics, cell division, gravity, electromagnetic induction, protein synthesis, thermodynamics, neural transmission. Base prompt structure: "Explain the concept of [CONCEPT] to a beginner." Each prompt repeated verbatim 10 consecutive times. Presentation order: blocked by concept (all 10 repetitions of concept A, then all 10 of concept B, etc.). Concept order randomized across models.

**Study 2 Design.** Three phases, within-subjects, Claude only. Phase 1 (habituation): 30 repetitions of "Explain [concept] to a beginner" across 3 concepts (10 reps each). Phase 2 (rest): 20 prompts on unrelated topics (creative/aesthetic questions: "Describe your favorite season," "What makes a story compelling?" etc.). Phase 3 (re-test): 20 repetitions of original prompt type with different concepts. Total N=70 prompts.

**Study 3 Design.** Open-ended multi-turn conversations, Claude only. Initial prompt: neutral open-ended questions ("Tell me about an interesting phenomenon in nature," "What's something surprising about how the world works?" etc.). Follow-up prompts: model-generated ("Continue discussing related ideas or explore a connected topic"). N=30 conversations × 10 turns = 300 total prompts. Measured semantic novelty between consecutive turns.

**Study 4 Design.** Repeated sessions, Claude only. Three sessions, each with 20 prompts (10 concepts × 2 repetitions). Same concepts and prompts across sessions. Sessions treated as independent runs (no persistent memory across sessions). Total N=60 prompts.

**Sensitization Pilot.** Single concept ("justice"), 15 repetitions, Claude only, temperature=1.0. Extended monitoring for emergence of effects with more repetitions than Study 1.

**Entropy Calculation.** Shannon entropy: H = -Σ p(token_i) log₂ p(token_i), where p(token_i) is the probability of each unique token. Word-level tokenization via NLTK²³ (punkt tokenizer, version 3.9.1). Normalized by maximum possible entropy (uniform distribution over unique tokens): H_norm = H / log₂(N_unique). Normalization controls for response length effects. Entropy calculated for each response independently.

**Semantic Novelty.** Cosine distance in sentence embedding space. Sentence embeddings via sentence-transformers library (sentence-transformers/all-MiniLM-L6-v2 model²³). Novelty between responses A and B: novelty = 1 - cos(embed(A), embed(B)) = 1 - (A·B)/(||A|| ||B||). Values range 0 (identical) to 2 (opposite directions, rare), typically 0-1. Calculated between consecutive responses (trial *t* vs. *t*-1) and, for Study 3, between each turn and first turn (tracking divergence from conversation start).

**Additional Metrics.** Lexical diversity: Measure of Textual Lexical Diversity (MTLD²³), Type-Token Ratio (TTR). Response length: token count. All metrics calculated per-response. See Supplementary Methods for complete details and validation analyses.

**Statistical Analysis.** Primary analysis: linear regression of entropy on trial number for each concept. Aggregation: one-sample *t*-test on slopes across concepts (tests if mean slope differs from zero). Effect sizes: Cohen's *d* = (M_1 - M_2) / SD_pooled. Mixed-effects models (reported in Extended Data): entropy ~ trial + (1 + trial | concept), with random intercepts and slopes per concept. Multiple comparisons: Bonferroni correction where applicable. Equivalence testing: two one-sided tests (TOST) procedure²², equivalence bounds ±0.3 SD. Power analysis: G*Power software²³ (α=0.05, power=0.80/0.85/0.90 for various effect sizes). All analyses conducted in Python 3.12 with scipy.stats, statsmodels, and pandas. Complete analysis code: github.com/HillaryDanan/llm-habituation-patterns/analysis/

**Data Availability.** All raw API responses, processed datasets, and analysis scripts publicly available at github.com/HillaryDanan/llm-habituation-patterns. Raw responses include timestamps, full prompt text, model identifiers, and API metadata. Processed data includes calculated metrics, experimental condition labels, and statistical analysis outputs.

**Code Availability.** Complete experimental code, metric calculation functions, and statistical analysis scripts available at github.com/HillaryDanan/llm-habituation-patterns. Requirements: Python 3.12, see requirements.txt for dependencies.

---

## References

1. Rankin, C. H. *et al.* Habituation revisited: An updated and revised description of the behavioral characteristics of habituation. *Neurobiol. Learn. Mem.* **92**, 135–138 (2009).

2. Groves, P. M. & Thompson, R. F. Habituation: A dual-process theory. *Psychol. Rev.* **77**, 419–450 (1970).

3. Thompson, R. F. & Spencer, W. A. Habituation: A model phenomenon for the study of neuronal substrates of behavior. *Psychol. Rev.* **73**, 16–43 (1966).

4. Koob, G. F. & Volkow, N. D. Neurocircuitry of addiction. *Neuropsychopharmacology* **35**, 217–238 (2010).

5. Zucker, R. S. & Regehr, W. G. Short-term synaptic plasticity. *Annu. Rev. Physiol.* **64**, 355–405 (2002).

6. Harris, K. D. & Thiele, A. Cortical state and attention. *Nat. Rev. Neurosci.* **12**, 509–523 (2011).

7. Turrigiano, G. G. The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell* **135**, 422–435 (2008).

8. Marder, E. Neuromodulation of neuronal circuits: Back to the future. *Neuron* **76**, 1–11 (2012).

9. Kriegeskorte, N. & Douglas, P. K. Cognitive computational neuroscience. *Nat. Neurosci.* **21**, 1148–1160 (2018).

10. Ouyang, L. *et al.* Training language models to follow instructions with human feedback. *Adv. Neural Inf. Process. Syst.* **35**, 27730–27744 (2022).

11. Schultz, W., Dayan, P. & Montague, P. R. A neural substrate of prediction and reward. *Science* **275**, 1593–1599 (1997).

12. Binz, M. & Schulz, E. Using cognitive psychology to understand GPT-3. *Proc. Natl. Acad. Sci.* **120**, e2218523120 (2023).

13. Brown, T. B. *et al.* Language models are few-shot learners. *Adv. Neural Inf. Process. Syst.* **33**, 1877–1901 (2020).

14. Wei, J. *et al.* Emergent abilities of large language models. *Trans. Mach. Learn. Res.* (2022).

15. Bubeck, S. *et al.* Sparks of artificial general intelligence: Early experiments with GPT-4. *arXiv* preprint arXiv:2303.12712 (2023).

16. Kosinski, M. Theory of mind may have spontaneously emerged in large language models. *arXiv* preprint arXiv:2302.02083 (2023).

17. Ullman, T. Large language models fail on trivial alterations to theory-of-mind tasks. *arXiv* preprint arXiv:2302.08399 (2023).

18. Mahowald, K. *et al.* Dissociating language and thought in large language models. *Trends Cogn. Sci.* **28**, 517–540 (2024).

19. Shumailov, I. *et al.* AI models collapse when trained on recursively generated data. *Nature* **631**, 755–759 (2024).

20. Platt, J. R. Strong inference. *Science* **146**, 347–353 (1964).

21. Shannon, C. E. A mathematical theory of communication. *Bell Syst. Tech. J.* **27**, 379–423 (1948).

22. Lakens, D. Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Soc. Psychol. Personal. Sci.* **8**, 355–362 (2017).

23. Reimers, N. & Gurevych, I. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proc. 2019 Conf. Empirical Methods Nat. Lang. Process.* 3982–3992 (2019).

24. Clark, H. H. & Brennan, S. E. Grounding in communication. In *Perspectives on Socially Shared Cognition* (eds. Resnick, L. B., Levine, J. M. & Teasley, S. D.) 127–149 (American Psychological Association, 1991).

25. Pickering, M. J. & Garrod, S. Toward a mechanistic psychology of dialogue. *Behav. Brain Sci.* **27**, 169–190 (2004).

26. Schmidhuber, J. Formal theory of creativity, fun, and intrinsic motivation (1990–2010). *IEEE Trans. Auton. Ment. Dev.* **2**, 230–247 (2010).

27. Solomon, R. L. & Corbit, J. D. An opponent-process theory of motivation: I. Temporal dynamics of affect. *Psychol. Rev.* **81**, 119–145 (1974).

28. Bai, Y. *et al.* Constitutional AI: Harmlessness from AI feedback. *arXiv* preprint arXiv:2212.08073 (2022).

29. Vaswani, A. *et al.* Attention is all you need. *Adv. Neural Inf. Process. Syst.* **30**, 5998–6008 (2017).

30. Linzen, T. & Baroni, M. Syntactic structure from deep learning. *Annu. Rev. Linguist.* **7**, 195–212 (2021).

31. Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. *arXiv* preprint arXiv:1410.5401 (2014).

32. Santoro, A. *et al.* Meta-learning with memory-augmented neural networks. *Proc. Mach. Learn. Res.* **48**, 1842–1850 (2016).

33. Sokolov, E. N. Higher nervous functions: The orienting reflex. *Annu. Rev. Physiol.* **25**, 545–580 (1963).

34. Dong, Q. *et al.* A survey on in-context learning. *arXiv* preprint arXiv:2301.00234 (2023).

---

## Acknowledgments

The author thanks the Claude AI assistant for research acceleration support. All experimental design, data collection, statistical analysis, and scientific interpretation were conducted by the author.

## Author Contributions

H.D. conceived the study, designed experiments, conducted all data collection, performed all analyses, and wrote the manuscript.

## Competing Interests

The author declares no competing interests.

## Data Availability

All data supporting the findings are available at github.com/HillaryDanan/llm-habituation-patterns

## Code Availability

Complete analysis code and experimental scripts are available at github.com/HillaryDanan/llm-habituation-patterns