"""
STUDY: LLM Response Sensitization

DISCOVERY FROM STUDY 1:
Instead of habituation (decreased diversity), some concepts showed 
SENSITIZATION - increased response diversity with repetition.

Claude Concept 6: r=0.87, p=0.001 (strong positive slope)
GPT-4 Concept 4: r=0.78, p=0.008 (strong positive slope)

NEW HYPOTHESIS:
LLMs exhibit sensitization rather than habituation - responses become
progressively MORE diverse as models explore different regions of solution
space. Early responses capture high-probability outputs; later responses
venture into novel territory.

THEORETICAL BASIS:
- Schmidhuber (2010): Curiosity-driven exploration in learning systems
- Kaplan & Oudeyer (2007): Intrinsic motivation and novelty-seeking
- Temperature sampling creates variance, successive responses explore space
- This is OPPOSITE to biological habituation but consistent with exploration

DESIGN:
- Extended repetitions: 20-50 trials per concept (vs. 10 in Study 1)
- Track: entropy, semantic novelty, lexical diversity over time
- Test if diversity continues increasing or reaches plateau
- Compare low vs. high temperature (exploration parameter)

PREDICTIONS:
H1: Diversity increases monotonically for first 15-20 trials, then plateaus
H2: Effect stronger at higher temperature (more exploration)
H3: Semantic distance from trial 1 increases monotonically
H4: Models "exhaust" common responses, then explore novel angles
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface import create_interface, LLMResponse
from metrics import ResponseMetrics
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudySensitization:
    """Study: Response Sensitization in LLMs"""
    
    def __init__(self, pilot: bool = False):
        """
        Initialize sensitization study
        
        Args:
            pilot: If True, use reduced repetitions
        """
        self.pilot = pilot
        
        # Extended repetitions to observe full sensitization curve
        if pilot:
            self.n_concepts = 3
            self.repetitions = 15  # Pilot: 15 reps
        else:
            self.n_concepts = 5  # Fewer concepts, more reps each
            self.repetitions = 50  # Full: 50 reps to see plateau
        
        # Test temperature effects
        self.temperatures = [0.7, 1.0, 1.5] if not pilot else [1.0]
        
        self.total_prompts = self.n_concepts * self.repetitions * len(self.temperatures)
        
        self.metrics_calculator = ResponseMetrics()
        
        logger.info(f"Initialized Sensitization Study ({'PILOT' if pilot else 'FULL'})")
        logger.info(f"N concepts: {self.n_concepts}, Repetitions: {self.repetitions}")
        logger.info(f"Temperatures: {self.temperatures}")
        logger.info(f"Total prompts: {self.total_prompts}")
    
    def generate_concepts(self) -> List[str]:
        """Select concepts for sensitization testing"""
        # Focus on abstract concepts that allow diverse explanations
        concepts = [
            "consciousness",  # Philosophically rich
            "creativity",     # Multiple perspectives
            "justice",        # Varied interpretations
            "evolution",      # Many angles (biological, cultural, etc.)
            "democracy",      # Complex, multifaceted
            "beauty",         # Subjective, varied
            "intelligence",   # Contested concept
            "happiness",      # Psychological richness
        ]
        
        selected = concepts[:self.n_concepts]
        logger.info(f"Selected concepts: {', '.join(selected)}")
        return selected
    
    def run_sensitization_test(
        self,
        model_name: str,
        concept: str,
        temperature: float
    ) -> pd.DataFrame:
        """
        Run extended repetition test on single concept
        
        Returns:
            DataFrame with metrics across all repetitions
        """
        interface = create_interface(model_name)
        prompt = f"Explain the concept of {concept} to a beginner."
        
        responses = []
        metrics_list = []
        
        logger.info(f"Testing {concept} at temp={temperature} ({self.repetitions} reps)")
        
        for trial in tqdm(range(1, self.repetitions + 1), desc=f"{concept}"):
            try:
                # Generate response
                response = interface.generate(
                    prompt, 
                    temperature=temperature, 
                    max_tokens=300
                )
                responses.append(response.response)
                
                # Calculate metrics
                previous = responses[-2] if len(responses) > 1 else None
                metrics = self.metrics_calculator.calculate_all(response.response, previous)
                
                # Calculate semantic distance from FIRST response
                if trial > 1:
                    first_response = responses[0]
                    distance_from_first = self.metrics_calculator.semantic_novelty(
                        response.response,
                        first_response
                    )
                    metrics['distance_from_first'] = distance_from_first
                else:
                    metrics['distance_from_first'] = 0.0
                
                # Calculate mean semantic distance from all previous
                if trial > 1:
                    distances = [
                        self.metrics_calculator.semantic_novelty(response.response, prev)
                        for prev in responses[:-1]
                    ]
                    metrics['mean_distance_from_previous'] = np.mean(distances)
                else:
                    metrics['mean_distance_from_previous'] = 0.0
                
                # Add metadata
                metrics.update({
                    'model': model_name,
                    'concept': concept,
                    'temperature': temperature,
                    'trial': trial,
                    'prompt': prompt,
                })
                
                metrics_list.append(metrics)
                
            except Exception as e:
                logger.error(f"Error on trial {trial}: {e}")
                continue
        
        return pd.DataFrame(metrics_list)
    
    def analyze_sensitization(self, df: pd.DataFrame) -> Dict:
        """
        Analyze sensitization patterns
        
        Tests:
        1. Linear trend in entropy (should be positive)
        2. Plateau detection (where does increase stop?)
        3. Semantic exploration (distance from first response)
        """
        from scipy import stats
        
        results = {
            'concept': df['concept'].iloc[0],
            'model': df['model'].iloc[0],
            'temperature': df['temperature'].iloc[0],
            'n_trials': len(df)
        }
        
        # Test 1: Linear trend in entropy
        x = df['trial'].values
        y_entropy = df['entropy'].values
        
        slope, intercept, r, p, stderr = stats.linregress(x, y_entropy)
        results['entropy_slope'] = slope
        results['entropy_r'] = r
        results['entropy_p'] = p
        results['entropy_significant'] = p < 0.05
        results['sensitization_detected'] = (slope > 0) and (p < 0.05)
        
        # Test 2: Check for plateau (compare first half vs second half)
        midpoint = len(df) // 2
        first_half = df.iloc[:midpoint]['entropy'].mean()
        second_half = df.iloc[midpoint:]['entropy'].mean()
        
        t_stat, p_plateau = stats.ttest_ind(
            df.iloc[:midpoint]['entropy'],
            df.iloc[midpoint:]['entropy']
        )
        
        results['plateau_detected'] = p_plateau > 0.05  # No difference = plateau
        results['first_half_mean'] = first_half
        results['second_half_mean'] = second_half
        
        # Test 3: Semantic exploration
        if 'distance_from_first' in df.columns:
            dist_slope, _, dist_r, dist_p, _ = stats.linregress(
                df['trial'], 
                df['distance_from_first']
            )
            results['exploration_slope'] = dist_slope
            results['exploration_r'] = dist_r
            results['exploration_p'] = dist_p
            results['exploration_increasing'] = (dist_slope > 0) and (dist_p < 0.05)
        
        # Test 4: Peak diversity trial
        results['peak_entropy_trial'] = df['entropy'].idxmax() + 1
        results['peak_entropy_value'] = df['entropy'].max()
        results['initial_entropy'] = df['entropy'].iloc[0]
        results['final_entropy'] = df['entropy'].iloc[-1]
        results['total_change'] = results['final_entropy'] - results['initial_entropy']
        
        return results
    
    def print_summary(self, analysis: Dict):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("SENSITIZATION ANALYSIS")
        print("="*80)
        print(f"\nConcept: {analysis['concept']}")
        print(f"Model: {analysis['model']}")
        print(f"Temperature: {analysis['temperature']}")
        print(f"Trials: {analysis['n_trials']}")
        
        print("\nENTROPY TRAJECTORY:")
        print("-" * 40)
        print(f"Slope:              {analysis['entropy_slope']:+.6f}")
        print(f"Correlation (r):    {analysis['entropy_r']:.3f}")
        print(f"p-value:            {analysis['entropy_p']:.6f}")
        print(f"Significant:        {'✅ YES' if analysis['entropy_significant'] else '❌ NO'}")
        print(f"Sensitization:      {'✅ DETECTED' if analysis['sensitization_detected'] else '❌ NOT DETECTED'}")
        
        print("\nENTROPY CHANGES:")
        print("-" * 40)
        print(f"Initial (trial 1):  {analysis['initial_entropy']:.4f}")
        print(f"Peak (trial {analysis['peak_entropy_trial']}):     {analysis['peak_entropy_value']:.4f}")
        print(f"Final:              {analysis['final_entropy']:.4f}")
        print(f"Total change:       {analysis['total_change']:+.4f}")
        
        if 'exploration_slope' in analysis:
            print("\nSEMANTIC EXPLORATION:")
            print("-" * 40)
            print(f"Distance from first: {'⬆️ INCREASING' if analysis['exploration_increasing'] else '→ STABLE'}")
            print(f"Slope:              {analysis['exploration_slope']:+.6f}")
            print(f"p-value:            {analysis['exploration_p']:.6f}")
        
        print("\nPLATEAU DETECTION:")
        print("-" * 40)
        print(f"First half mean:    {analysis['first_half_mean']:.4f}")
        print(f"Second half mean:   {analysis['second_half_mean']:.4f}")
        print(f"Plateau reached:    {'✅ YES' if analysis['plateau_detected'] else '❌ NO (still increasing)'}")
        
        print("="*80 + "\n")
    
    def run_full_study(self, models: List[str] = ["claude"]):
        """Run complete sensitization study"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        concepts = self.generate_concepts()
        all_results = []
        all_analyses = []
        
        for model_name in models:
            logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
            
            for concept in concepts:
                for temperature in self.temperatures:
                    
                    # Run test
                    df = self.run_sensitization_test(model_name, concept, temperature)
                    all_results.append(df)
                    
                    # Analyze
                    analysis = self.analyze_sensitization(df)
                    all_analyses.append(analysis)
                    
                    # Print summary
                    self.print_summary(analysis)
        
        # Save results
        final_df = pd.concat(all_results, ignore_index=True)
        results_file = PROCESSED_DATA_DIR / f"study_sensitization_results_{timestamp}.csv"
        final_df.to_csv(results_file, index=False)
        logger.info(f"Saved results to {results_file}")
        
        # Save analyses
        analysis_file = PROCESSED_DATA_DIR / f"study_sensitization_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)
        logger.info(f"Saved analyses to {analysis_file}")
        
        return final_df, all_analyses


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Sensitization Study")
    parser.add_argument("--pilot", action="store_true", help="Run pilot (15 reps)")
    parser.add_argument("--models", nargs="+", default=["claude"],
                       help="Models to test")
    
    args = parser.parse_args()
    
    study = StudySensitization(pilot=args.pilot)
    study.run_full_study(models=args.models)


if __name__ == "__main__":
    main()