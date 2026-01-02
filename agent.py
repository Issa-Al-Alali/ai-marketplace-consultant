import google.generativeai as genai
import json
import time
from prompts import *

class MarketplaceAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.call_count = 0
        self.total_tokens = 0
        
    def call_llm(self, prompt, parse_json=True):
        """Make LLM call with error handling and tracking"""
        self.call_count += 1
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            if parse_json:
                # Extract JSON from markdown code blocks if present
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Fallback: return as text if JSON parsing fails
                    return {"raw_response": text, "parse_error": True}
            return text
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_chain_of_thought(self, metrics):
        """Basic CoT approach with step-by-step reasoning"""
        prompt = COT_PROMPT.format(metrics=json.dumps(metrics, indent=2))
        return self.call_llm(prompt)
    
    def run_few_shot_cot(self, metrics):
        """CoT with example demonstrations"""
        prompt = FEW_SHOT_COT_PROMPT.format(metrics=json.dumps(metrics, indent=2))
        return self.call_llm(prompt)
    
    def run_evaluator_optimizer(self, metrics):
        """Two-stage: Draft → Critique → Refine"""
        draft = self.run_chain_of_thought(metrics)
        
        critique_prompt = EVALUATOR_PROMPT.format(
            metrics=json.dumps(metrics, indent=2),
            draft=json.dumps(draft, indent=2)
        )
        critique = self.call_llm(critique_prompt)
        
        optimizer_prompt = OPTIMIZER_PROMPT.format(
            metrics=json.dumps(metrics, indent=2),
            draft=json.dumps(draft, indent=2),
            critique=json.dumps(critique, indent=2)
        )
        refined = self.call_llm(optimizer_prompt)
        
        return {
            "method": "evaluator_optimizer",
            "draft": draft,
            "critique": critique,
            "final": refined
        }
    
    def run_voting(self, metrics, num_voters=3):
        """Generate multiple analyses and vote on best"""
        results = []
        
        for i in range(num_voters):
            result = self.run_chain_of_thought(metrics)
            results.append(result)
            time.sleep(0.5)
        
        voting_prompt = VOTING_PROMPT.format(
            metrics=json.dumps(metrics, indent=2),
            result1=json.dumps(results[0], indent=2),
            result2=json.dumps(results[1], indent=2),
            result3=json.dumps(results[2], indent=2)
        )
        consensus = self.call_llm(voting_prompt)
        
        return {
            "method": "voting",
            "individual_results": results,
            "consensus": consensus
        }
    
    def run_self_consistency(self, metrics):
        """Multiple reasoning paths → converge to consistent answer"""
        prompt = SELF_CONSISTENCY_PROMPT.format(
            metrics=json.dumps(metrics, indent=2)
        )
        return self.call_llm(prompt)
    
    def run_comparative_analysis(self, metrics, benchmarks):
        """Analysis with peer comparison context"""
        prompt = COMPARATIVE_PROMPT.format(
            metrics=json.dumps(metrics, indent=2),
            benchmarks=json.dumps(benchmarks, indent=2)
        )
        return self.call_llm(prompt)
    
    def batch_analyze(self, vendor_list, method='chain_of_thought', progress_callback=None):
        """Process multiple vendors with progress tracking"""
        results = []
        total = len(vendor_list)
        
        for idx, vendor_metrics in enumerate(vendor_list):
            if progress_callback:
                progress_callback(idx, total)
            
            if method == 'chain_of_thought':
                result = self.run_chain_of_thought(vendor_metrics)
            elif method == 'few_shot':
                result = self.run_few_shot_cot(vendor_metrics)
            elif method == 'evaluator_optimizer':
                result = self.run_evaluator_optimizer(vendor_metrics)
            elif method == 'voting':
                result = self.run_voting(vendor_metrics)
            elif method == 'self_consistency':
                result = self.run_self_consistency(vendor_metrics)
            else:
                result = {"error": "Unknown method"}
            
            results.append({
                'seller_id': vendor_metrics.get('seller_id'),
                'analysis': result
            })
            
            time.sleep(0.5)
        
        return results
    
    def get_stats(self):
        """Return agent statistics"""
        return {
            'total_llm_calls': self.call_count,
            'estimated_tokens': self.total_tokens
        }