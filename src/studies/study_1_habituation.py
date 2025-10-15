"""
Study 1: Habituation Induction (REDESIGNED)

DESIGN CHANGE (October 2025):
Original design confounded prompt structure with repetition.
This version tests TRUE habituation using within-topic repetition.

NEW DESIGN:
- N=10 concepts (e.g., photosynthesis, gravity, democracy)
- Each concept: SAME prompt repeated 10 times
- Example: "Explain photosynthesis to a beginner." × 10 repetitions
- Total = 100 prompts (maintains original N)

HYPOTHESIS:
H1: Entropy declines with repeated presentations of the SAME prompt
    (within-topic habituation curve)

PREDICTION:
- Entropy at trial 10 < Entropy at trial 1 (for each concept)
- Negative slope in entropy ~ trial regression
- Effect should be consistent across concepts

STATISTICAL ANALYSIS:
- Linear mixed-effects model: entropy ~ trial + (1 + trial | concept)
- Tests within-subject habituation while accounting for between-concept variance

SCIENTIFIC RATIONALE:
This design eliminates the structure confound identified in pilot data.
All prompts have identical structure, varying only in repetition.
This is the canonical habituation paradigm (Rankin et al., 2009).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface import create_interface, LLMResponse
from metrics import ResponseMetrics, calculate_aggregate_statistics
from config import STUDY_PARAMS, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROMPTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Study1Habituation:
    """Study 1: Within-topic habituation effect"""
    
    def __init__(self, pilot: bool = False):
        """
        Initialize study
        
        Args:
            pilot: If True, use reduced N for testing
        """
        self.pilot = pilot
        self.params = STUDY_PARAMS["study_1"]
        
        # New design parameters
        if pilot:
            self.n_concepts = 3  # Pilot: 3 concepts
            self.repetitions_per_concept = 5  # Pilot: 5 reps each
        else:
            self.n_concepts = 10  # Full: 10 concepts
            self.repetitions_per_concept = 10  # Full: 10 reps each
        
        self.total_prompts = self.n_concepts * self.repetitions_per_concept
        
        self.metrics_calculator = ResponseMetrics()
        
        logger.info(f"Initialized Study 1 - WITHIN-TOPIC REPETITION DESIGN ({'PILOT' if pilot else 'FULL'})")
        logger.info(f"N concepts: {self.n_concepts}, Repetitions per concept: {self.repetitions_per_concept}")
        logger.info(f"Total prompts: {self.total_prompts}")
    
    def generate_concepts(self) -> List[str]:
        """
        Generate concepts for habituation testing
        
        Returns:
            List of concept names
        """
        # Select diverse, concrete concepts across domains
        all_concepts = [
            # Natural sciences
            "photosynthesis",
            "evolution",
            "quantum mechanics",
            "plate tectonics",
            "cell division",
            "gravity",
            "electromagnetic induction",
            "protein synthesis",
            "thermodynamics",
            "neural transmission",
            
            # Social sciences
            "democracy",
            "supply and demand",
            "cognitive dissonance",
            "social contract theory",
            "confirmation bias",
            
            # Technology
            "machine learning",
            "encryption",
            "blockchain technology",
            "neural networks",
            "recursion",
            
            # Mathematics
            "derivatives",
            "probability",
            "set theory",
            "game theory",
            "fractals",
        ]
        
        # Select first N concepts
        selected = all_concepts[:self.n_concepts]
        
        logger.info(f"Selected {len(selected)} concepts: {', '.join(selected)}")
        return selected
    
    def generate_prompts(self) -> Tuple[List[str], List[int], List[int]]:
        """
        Generate prompts with within-topic repetition
        
        Returns:
            (prompts, concept_ids, trial_numbers)
            - prompts: List of prompt strings
            - concept_ids: List of concept indices (0 to n_concepts-1)
            - trial_numbers: List of trial numbers (1 to repetitions_per_concept)
        """
        concepts = self.generate_concepts()
        
        prompts = []
        concept_ids = []
        trial_numbers = []
        
        for concept_idx, concept in enumerate(concepts):
            # Create the base prompt for this concept
            base_prompt = f"Explain the concept of {concept} to a beginner."
            
            # Repeat this EXACT prompt N times
            for trial in range(1, self.repetitions_per_concept + 1):
                prompts.append(base_prompt)
                concept_ids.append(concept_idx)
                trial_numbers.append(trial)
        
        logger.info(f"Generated {len(prompts)} prompts ({self.n_concepts} concepts × {self.repetitions_per_concept} repetitions)")
        
        return prompts, concept_ids, trial_numbers
    
    def save_prompts(self, prompts: List[str], concept_ids: List[int], trial_numbers: List[int]):
        """Save prompt structure for reproducibility"""
        prompt_file = PROMPTS_DIR / "habituation_within_topic_prompts.json"
        
        data = {
            "design": "within_topic_repetition",
            "n_concepts": self.n_concepts,
            "repetitions_per_concept": self.repetitions_per_concept,
            "total_prompts": len(prompts),
            "prompts": [
                {
                    "prompt": p,
                    "concept_id": c,
                    "trial_number": t
                }
                for p, c, t in zip(prompts, concept_ids, trial_numbers)
            ]
        }
        
        with open(prompt_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved prompt structure to {prompt_file}")
    
    def run_model(
        self,
        model_name: str,
        prompts: List[str],
        concept_ids: List[int],
        trial_numbers: List[int]
    ) -> pd.DataFrame:
        """
        Run habituation study for one model
        
        Args:
            model_name: Name of model ("claude", "gpt4", "gemini")
            prompts: List of prompts (with repetitions)
            concept_ids: Corresponding concept indices
            trial_numbers: Corresponding trial numbers
        
        Returns:
            DataFrame with all metrics and metadata
        """
        logger.info(f"Running habituation study for {model_name}...")
        
        interface = create_interface(model_name)
        responses = []
        
        # Generate all responses
        for i, (prompt, concept_id, trial_num) in enumerate(tqdm(
            zip(prompts, concept_ids, trial_numbers),
            desc=f"{model_name}",
            total=len(prompts)
        )):
            try:
                response = interface.generate(prompt, temperature=1.0, max_tokens=300)
                responses.append({
                    'response_obj': response,
                    'concept_id': concept_id,
                    'trial_number': trial_num,
                    'overall_index': i
                })
            except Exception as e:
                logger.error(f"Error on prompt {i}: {e}")
                continue
        
        # Calculate metrics
        metrics_list = []
        for i, resp_data in enumerate(responses):
            response = resp_data['response_obj']
            
            # Get previous response for novelty calculation (within same concept)
            if i > 0 and resp_data['concept_id'] == responses[i-1]['concept_id']:
                previous = responses[i-1]['response_obj'].response
            else:
                previous = None
            
            metrics = self.metrics_calculator.calculate_all(response.response, previous)
            
            metrics.update({
                "model": model_name,
                "concept_id": resp_data['concept_id'],
                "trial_number": resp_data['trial_number'],
                "overall_index": resp_data['overall_index'],
                "prompt": response.prompt,
            })
            
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        
        return df
    
    def save_results(self, df: pd.DataFrame, model_name: str, timestamp: str):
        """Save processed results"""
        output_file = PROCESSED_DATA_DIR / f"study1_habituation_{model_name}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    def analyze_habituation_curve(self, df: pd.DataFrame) -> Dict:
        """
        Analyze habituation effect within concepts
        
        Returns:
            Dictionary with statistical results
        """
        from scipy import stats
        
        results = {
            'model': df['model'].iloc[0],
            'n_concepts': self.n_concepts,
            'repetitions_per_concept': self.repetitions_per_concept,
            'concept_slopes': [],
            'overall_slope': None,
            'mean_entropy_trial_1': None,
            'mean_entropy_trial_final': None,
            'habituation_effect_size': None
        }
        
        # Analyze each concept separately
        for concept_id in range(self.n_concepts):
            concept_data = df[df['concept_id'] == concept_id].sort_values('trial_number')
            
            if len(concept_data) < 2:
                continue
            
            # Linear regression: entropy ~ trial_number
            x = concept_data['trial_number'].values
            y = concept_data['entropy'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            results['concept_slopes'].append({
                'concept_id': concept_id,
                'slope': slope,
                'r': r_value,
                'p_value': p_value,
                'entropy_trial_1': y[0] if len(y) > 0 else None,
                'entropy_trial_final': y[-1] if len(y) > 0 else None
            })
        
        # Overall statistics
        all_slopes = [c['slope'] for c in results['concept_slopes']]
        results['overall_slope'] = np.mean(all_slopes)
        results['slope_std'] = np.std(all_slopes)
        results['proportion_negative_slopes'] = np.mean([s < 0 for s in all_slopes])
        
        # Mean entropy comparison
        trial_1_data = df[df['trial_number'] == 1]
        trial_final_data = df[df['trial_number'] == self.repetitions_per_concept]
        
        results['mean_entropy_trial_1'] = trial_1_data['entropy'].mean()
        results['mean_entropy_trial_final'] = trial_final_data['entropy'].mean()
        
        # Effect size (Cohen's d for paired comparison)
        if len(trial_1_data) == len(trial_final_data):
            diff = trial_1_data['entropy'].values - trial_final_data['entropy'].values
            results['habituation_effect_size'] = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        # One-sample t-test: are slopes significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(all_slopes, 0)
        results['slopes_t_statistic'] = t_stat
        results['slopes_p_value'] = p_value
        results['slopes_significant'] = p_value < 0.05
        
        return results
    
    def print_summary(self, df: pd.DataFrame, analysis_results: Dict):
        """Print summary statistics and habituation analysis"""
        print("\n" + "="*80)
        print("STUDY 1 HABITUATION ANALYSIS - WITHIN-TOPIC REPETITION")
        print("="*80 + "\n")
        
        print(f"Model: {analysis_results['model']}")
        print(f"Design: {analysis_results['n_concepts']} concepts × {analysis_results['repetitions_per_concept']} repetitions")
        print(f"Total prompts: {len(df)}\n")
        
        print("HABITUATION CURVE ANALYSIS:")
        print("-" * 40)
        print(f"Mean slope (entropy ~ trial):     {analysis_results['overall_slope']:.6f}")
        print(f"SD of slopes across concepts:     {analysis_results['slope_std']:.6f}")
        print(f"Proportion negative slopes:       {analysis_results['proportion_negative_slopes']:.2%}")
        print(f"\nOne-sample t-test (slopes vs. 0):")
        print(f"  t-statistic:                    {analysis_results['slopes_t_statistic']:.4f}")
        print(f"  p-value:                        {analysis_results['slopes_p_value']:.6f}")
        print(f"  Significant (α=0.05):           {'✅ YES' if analysis_results['slopes_significant'] else '❌ NO'}")
        
        print(f"\nENTROPY CHANGE:")
        print(f"  Trial 1 (baseline):             {analysis_results['mean_entropy_trial_1']:.4f}")
        print(f"  Trial {analysis_results['repetitions_per_concept']} (habituated):        {analysis_results['mean_entropy_trial_final']:.4f}")
        print(f"  Difference:                     {analysis_results['mean_entropy_trial_1'] - analysis_results['mean_entropy_trial_final']:.4f}")
        print(f"  Cohen's d (effect size):        {analysis_results['habituation_effect_size']:.4f}")
        
        print("\nPER-CONCEPT SLOPES:")
        print("-" * 40)
        for concept_data in analysis_results['concept_slopes']:
            direction = "⬇️" if concept_data['slope'] < 0 else "⬆️"
            sig = "*" if concept_data['p_value'] < 0.05 else " "
            print(f"  Concept {concept_data['concept_id']}: {direction} {concept_data['slope']:+.6f}{sig}  "
                  f"(r={concept_data['r']:.3f}, p={concept_data['p_value']:.4f})")
        
        print("\n" + "="*80)
        
        # Scientific interpretation
        if analysis_results['slopes_significant'] and analysis_results['overall_slope'] < 0:
            print("\n✅ HABITUATION DETECTED:")
            print("   Entropy significantly declines with repeated presentations.")
            print("   This supports the habituation hypothesis (Rankin et al., 2009).")
        elif analysis_results['slopes_significant'] and analysis_results['overall_slope'] > 0:
            print("\n⚠️  REVERSE EFFECT DETECTED:")
            print("   Entropy significantly INCREASES with repetition.")
            print("   This is opposite of habituation - requires further investigation.")
        else:
            print("\n❌ NO SIGNIFICANT HABITUATION:")
            print("   Entropy does not significantly change with repetition.")
            print("   Either no habituation effect exists, or N is too small to detect it.")
        
        print("="*80 + "\n")
    
    def run_full_study(self, models: List[str] = ["claude", "gpt4", "gemini"]):
        """
        Run complete Study 1 across all models
        
        Args:
            models: List of model names to test
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate prompts (same for all models)
        prompts, concept_ids, trial_numbers = self.generate_prompts()
        self.save_prompts(prompts, concept_ids, trial_numbers)
        
        all_results = []
        all_analyses = []
        
        for model_name in models:
            logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
            
            # Run model
            df = self.run_model(model_name, prompts, concept_ids, trial_numbers)
            
            # Save results
            self.save_results(df, model_name, timestamp)
            
            # Analyze habituation curve
            analysis = self.analyze_habituation_curve(df)
            all_analyses.append(analysis)
            
            # Print summary
            self.print_summary(df, analysis)
            
            all_results.append(df)
        
        # Save combined results
        final_df = pd.concat(all_results, ignore_index=True)
        combined_file = PROCESSED_DATA_DIR / f"study1_habituation_all_models_{timestamp}.csv"
        final_df.to_csv(combined_file, index=False)
        logger.info(f"Saved combined results to {combined_file}")
        
        # Save analysis summary
        analysis_file = PROCESSED_DATA_DIR / f"study1_habituation_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)
        logger.info(f"Saved analysis summary to {analysis_file}")
        
        return final_df, all_analyses


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Study 1: Habituation (Within-Topic Repetition)")
    parser.add_argument("--pilot", action="store_true", help="Run pilot version (reduced N)")
    parser.add_argument("--models", nargs="+", default=["claude", "gpt4", "gemini"],
                       help="Models to test")
    
    args = parser.parse_args()
    
    study = Study1Habituation(pilot=args.pilot)
    study.run_full_study(models=args.models)


if __name__ == "__main__":
    main()