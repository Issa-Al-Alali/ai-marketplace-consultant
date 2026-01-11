# import google.generativeai as genai
import json
import time
from prompts import *
from groq import Groq

class MarketplaceAgent:
    def __init__(self, api_key):
        # ===== GEMINI INITIALIZATION (COMMENTED OUT) =====
        # genai.configure(api_key=api_key)
        # self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # ===== GROQ INITIALIZATION =====
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"  # You can change to other Groq models like "mixtral-8x7b-32768" or "llama-3.1-70b-versatile"
        
        self.call_count = 0
        self.total_tokens = 0
    
    def _correct_tier_from_score(self, result):
        """Post-process to ensure tier matches score thresholds: C=0-40, B=40-70, A=70-100"""
        if not isinstance(result, dict):
            return result
        
        score = result.get('overall_score') or result.get('score')
        if score is not None:
            try:
                score = float(score)
                if score >= 70:
                    result['tier'] = 'A'
                elif score >= 40:
                    result['tier'] = 'B'
                else:
                    result['tier'] = 'C'
            except (ValueError, TypeError):
                pass
        
        return result
        
    def call_llm(self, prompt, parse_json=True, max_retries=2):
        """Make LLM call with error handling and tracking"""
        self.call_count += 1
        
        for attempt in range(max_retries + 1):
            try:
                # Add system prompt context for JSON output
                # ===== GEMINI API CALL (COMMENTED OUT) =====
                # full_prompt = prompt
                # if parse_json:
                #     full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nIMPORTANT: You must respond with ONLY valid JSON, no additional text or explanation."
                # response = self.model.generate_content(full_prompt)
                # text = response.text
                
                # ===== GROQ API CALL =====
                user_content = prompt
                if parse_json:
                    user_content = f"{prompt}\n\nIMPORTANT: You must respond with ONLY valid JSON, no additional text or explanation."
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT if parse_json else "You are a helpful assistant."},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                text = response.choices[0].message.content
                
                # Track tokens if available
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
                
                if parse_json:
                    # Extract JSON from markdown code blocks if present
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        # Extract from first code block
                        parts = text.split("```")
                        if len(parts) >= 2:
                            text = parts[1].strip()
                            # Remove language identifier if present
                            if text.startswith("json"):
                                text = text[4:].strip()
                    
                    # Try to find JSON object in text
                    try:
                        result = json.loads(text)
                        # Validate result has required fields
                        if not isinstance(result, dict):
                            raise ValueError("Result is not a dictionary")
                        return result
                    except (json.JSONDecodeError, ValueError):
                        # Try to extract JSON object from text
                        import re
                        json_match = re.search(r'\{.*\}', text, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(0))
                                if isinstance(result, dict):
                                    return result
                            except json.JSONDecodeError:
                                pass
                        
                        # Only fail on last attempt
                        if attempt < max_retries:
                            time.sleep(1)  # Brief wait before retry
                            continue
                        
                        # Fallback: return as text if JSON parsing fails
                        if attempt == max_retries:  # Only print on final attempt
                            print(f"WARNING: Failed to parse JSON after {max_retries + 1} attempts. Response preview: {text[:200]}")
                        return {"raw_response": text[:500], "parse_error": True, "error": "JSON parsing failed"}
                return text
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries:
                        # Extract wait time if available
                        import re
                        wait_match = re.search(r'retry in ([\d.]+)', error_msg, re.IGNORECASE)
                        wait_time = float(wait_match.group(1)) if wait_match else 2.0
                        print(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"WARNING: Rate limit error after {max_retries + 1} attempts: {error_msg[:200]}")
                
                # Return error on final attempt
                if attempt == max_retries:
                    return {"error": error_msg, "tier": None, "overall_score": None}
        
        # Should never reach here, but just in case
        return {"error": "Unexpected error in call_llm", "tier": None, "overall_score": None}
    
    def run_chain_of_thought(self, metrics):
        """Basic CoT approach with step-by-step reasoning"""
        prompt = COT_PROMPT.format(metrics=json.dumps(metrics, indent=2))
        result = self.call_llm(prompt)
        return self._correct_tier_from_score(result)
    
    def run_few_shot_cot(self, metrics):
        """CoT with example demonstrations"""
        prompt = FEW_SHOT_COT_PROMPT.format(metrics=json.dumps(metrics, indent=2))
        result = self.call_llm(prompt)
        return self._correct_tier_from_score(result)
    
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
        refined = self._correct_tier_from_score(refined)
        
        return {
            "method": "evaluator_optimizer",
            "draft": self._correct_tier_from_score(draft),
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
        consensus = self._correct_tier_from_score(consensus)
        
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
        result = self.call_llm(prompt)
        return self._correct_tier_from_score(result)
    
    def run_comparative_analysis(self, metrics, benchmarks):
        """Analysis with peer comparison context"""
        prompt = COMPARATIVE_PROMPT.format(
            metrics=json.dumps(metrics, indent=2),
            benchmarks=json.dumps(benchmarks, indent=2)
        )
        result = self.call_llm(prompt)
        return self._correct_tier_from_score(result)
    
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