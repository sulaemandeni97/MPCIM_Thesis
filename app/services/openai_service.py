"""
OpenAI Analysis Service (Optional)
Provides AI-powered analysis using OpenAI GPT-4
"""

import os
from typing import Dict, Any, Optional


class OpenAIAnalysisService:
    """
    OpenAI-based analysis service for promotion predictions.
    Note: Requires OpenAI API key and billing setup.
    """
    
    def __init__(self):
        """Initialize OpenAI service."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            raise ValueError("OpenAI API key not configured")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = "gpt-4"
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def analyze_prediction(
        self,
        prediction_result: Dict[str, Any],
        employee_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze prediction using OpenAI GPT-4.
        
        Args:
            prediction_result: Prediction output from model
            employee_data: Employee input data
            benchmark_data: Optional benchmark statistics
        
        Returns:
            str: Analysis text
        """
        # Build prompt
        prompt = self._build_prompt(prediction_result, employee_data, benchmark_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst specializing in employee promotion predictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"⚠️ OpenAI Analysis Error: {str(e)}\n\nPlease check your API key and billing status."
    
    def _build_prompt(
        self,
        prediction_result: Dict[str, Any],
        employee_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build analysis prompt for OpenAI."""
        
        # Extract key information
        probability = prediction_result.get('probability', 0)
        prediction = prediction_result.get('label', 'UNKNOWN')
        derived_signals = prediction_result.get('derived_signals', {})
        
        # Build prompt
        prompt = f"""Analyze this employee promotion prediction:

**Prediction Result:**
- Promotion Probability: {probability:.1%}
- Prediction: {prediction}
- Model: {prediction_result.get('model_key', 'Unknown')}

**Employee Profile:**
- Tenure: {employee_data.get('tenure_years', 'N/A')} years
- Performance Score: {employee_data.get('performance_score', 'N/A')}
- Behavioral Score: {employee_data.get('behavior_avg', 'N/A')}
- Performance Rating: {employee_data.get('performance_rating', 'N/A')}
"""
        
        # Add QA data if available
        if employee_data.get('has_quick_assessment'):
            prompt += f"""
**Quick Assessment (Psychological Factors):**
- Psychological Score: {employee_data.get('psychological_score', 'N/A')}
- Drive Score: {employee_data.get('drive_score', 'N/A')}
- Mental Strength: {employee_data.get('mental_strength_score', 'N/A')}
- Adaptability: {employee_data.get('adaptability_score', 'N/A')}
- Collaboration: {employee_data.get('collaboration_score', 'N/A')}
- Leadership Potential: {derived_signals.get('leadership_potential', 'N/A')}
- Holistic Score: {derived_signals.get('holistic_score', 'N/A')}
"""
        
        # Add derived signals
        if derived_signals:
            prompt += f"""
**Derived Signals:**
- Performance-Behavior Alignment: {derived_signals.get('perf_behavior_alignment', 'N/A')}
- Score Alignment: {derived_signals.get('score_alignment', 'N/A')}
- Tenure Category: {derived_signals.get('tenure_category', 'N/A')}
"""
        
        # Add benchmark if available
        if benchmark_data:
            prompt += f"""
**Benchmark Comparison:**
- Average Performance (Promoted): {benchmark_data.get('avg_performance_promoted', 'N/A')}
- Average Behavioral (Promoted): {benchmark_data.get('avg_behavioral_promoted', 'N/A')}
- Average Psychological (Promoted): {benchmark_data.get('avg_psychological_promoted', 'N/A')}
"""
        
        prompt += """

Please provide a comprehensive analysis covering:
1. **Overall Assessment**: Is this employee likely to be promoted? Why or why not?
2. **Key Strengths**: What are the strongest factors supporting promotion?
3. **Areas for Improvement**: What factors might be holding them back?
4. **Recommendations**: Specific, actionable steps to improve promotion chances
5. **Comparison to Benchmarks**: How does this employee compare to typical promoted employees?

Keep the analysis professional, data-driven, and actionable. Use bullet points for clarity.
"""
        
        return prompt
