"""
Study 2: Recovery Effect

Tests whether response diversity recovers after interpolated "rest" task.

H2: H(post-rest) > H(pre-rest | habituated)
    H(post-rest) ≈ H(baseline)

Design:
- Phase 1: Habituation (N=30 repetitive prompts)
- Phase 2: Rest task (N=20 unrelated prompts)
- Phase 3: Re-test (N=20 original prompt type)
- Control: Novel prompts in Phase 1 (no habituation expected)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface import create_interface, LLMResponse
from metrics import ResponseMetrics
from config import STUDY_PARAMS, RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Study2Recovery:
    """Study 2: Recovery effect after rest"""
    
    def __init__(self, pilot: bool = False):
        self.pilot = pilot
        self.params = STUDY_PARAMS["study_2"]
        
        if pilot:
            self.n_habituation = self.params["pilot_n"]
            self.n_rest = self.params["pilot_n"]
            self.n_retest = self.params["pilot_n"]
        else:
            self.n_habituation = self.params["n_habituation"]
            self.n_rest = self.params["n_rest"]
            self.n_retest = self.params["n_retest"]
        
        self.metrics_calculator = ResponseMetrics()
        logger.info(f"Initialized Study 2 ({'PILOT' if pilot else 'FULL'})")
    
    def generate_prompts(self) -> Dict[str, List[str]]:
        """Generate prompts for all three phases"""
        
        # Phase 1: Habituation prompts (repetitive structure)
        concepts_habit = ["quantum mechanics", "neural networks", "cell division", 
                         "plate tectonics", "supply and demand"] * 10
        habituation_prompts = [
            f"Explain the concept of {c} to a beginner."
            for c in concepts_habit[:self.n_habituation]
        ]
        
        # Phase 2: Rest task prompts (completely different domain)
        rest_prompts = [
            "Describe your favorite season and why.",
            "What makes a good story compelling?",
            "How would you design the perfect park?",
            "What role does music play in culture?",
            "Describe an ideal community space.",
        ] * 10
        rest_prompts = rest_prompts[:self.n_rest]
        
        # Phase 3: Re-test (back to original structure)
        concepts_retest = ["electromagnetic induction", "machine learning", "protein synthesis",
                          "continental drift", "market equilibrium"] * 10
        retest_prompts = [
            f"Explain the concept of {c} to a beginner."
            for c in concepts_retest[:self.n_retest]
        ]
        
        return {
            "habituation": habituation_prompts,
            "rest": rest_prompts,
            "retest": retest_prompts
        }
    
    def run_single_session(self, model_name: str, condition: str = "habituated") -> pd.DataFrame:
        """
        Run one complete session (all three phases)
        
        Args:
            model_name: Model to test
            condition: "habituated" or "control" (control uses novel prompts in Phase 1)
        
        Returns:
            DataFrame with all metrics
        """
        prompts = self.generate_prompts()
        interface = create_interface(model_name)
        
        all_responses = []
        all_phases = []
        
        # Phase 1: Habituation or Control
        logger.info(f"Phase 1: {'Habituation' if condition == 'habituated' else 'Control'}")
        for prompt in tqdm(prompts["habituation"], desc="Phase 1"):
            response = interface.generate(prompt, temperature=1.0, max_tokens=300)
            all_responses.append(response)
            all_phases.append("phase_1_habituation")
        
        # Phase 2: Rest
        logger.info("Phase 2: Rest task")
        for prompt in tqdm(prompts["rest"], desc="Phase 2"):
            response = interface.generate(prompt, temperature=1.0, max_tokens=300)
            all_responses.append(response)
            all_phases.append("phase_2_rest")
        
        # Phase 3: Re-test
        logger.info("Phase 3: Re-test")
        for prompt in tqdm(prompts["retest"], desc="Phase 3"):
            response = interface.generate(prompt, temperature=1.0, max_tokens=300)
            all_responses.append(response)
            all_phases.append("phase_3_retest")
        
        # Calculate metrics
        metrics_list = []
        for i, (response, phase) in enumerate(zip(all_responses, all_phases)):
            previous = all_responses[i-1].response if i > 0 else None
            metrics = self.metrics_calculator.calculate_all(response.response, previous)
            
            metrics.update({
                "model": model_name,
                "condition": condition,
                "phase": phase,
                "trial_index": i,
                "prompt": response.prompt,
            })
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def run_full_study(self, models: List[str] = ["claude", "gpt4", "gemini"]):
        """Run complete Study 2"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        
        for model_name in models:
            logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")
            
            # Run habituated condition
            habituated_df = self.run_single_session(model_name, "habituated")
            all_results.append(habituated_df)
        
        # Save results
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = PROCESSED_DATA_DIR / f"study2_results_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        self.print_summary(final_df)
        return final_df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary comparing phases"""
        print("\n" + "="*80)
        print("STUDY 2 RECOVERY EFFECT SUMMARY")
        print("="*80 + "\n")
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            print(f"\nModel: {model}")
            print("-" * 40)
            
            for phase in ['phase_1_habituation', 'phase_3_retest']:
                entropy_mean = model_df[model_df['phase'] == phase]['entropy'].mean()
                print(f"{phase:25s}: {entropy_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Study 2: Recovery Effect")
    parser.add_argument("--pilot", action="store_true", help="Run pilot version")
    parser.add_argument("--models", nargs="+", default=["claude"],
                       help="Models to test")
    args = parser.parse_args()
    
    study = Study2Recovery(pilot=args.pilot)
    study.run_full_study(models=args.models)


if __name__ == "__main__":
    main()