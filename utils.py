from llama_cpp import Llama
import pandas as pd
from typing import Dict, Union
import json
import re


class RiskEstimateProcessor:
    def __init__(self, model_path: str):
        """Initialize with a Llama model."""
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4
        )

    def parse_estimate(self, row: pd.Series) -> Dict[str, Union[float, str, Dict]]:
        """
        Parse a single risk estimate using LLM.

        Parameters:
        row: Series containing estimate information

        Returns:
        Dictionary with parsed information
        """
        prompt = f"""
        Task: Analyze this existential risk estimate and convert it to a per-century probability.

        Estimate Information:
        - Original estimate: {row['estimation']}
        - Estimation description: {row['estimation_measure']}
        - Date made: {row['date']}
        - Estimator: {row['estimator']}
        - Other remarks: {row['other_notes']}

        Please analyze this information and provide:
        1. The per-century probability (as a decimal between 0 and 1)
        2. The reasoning behind the conversion
        3. Confidence in the conversion (high/medium/low)

        Output the result in JSON format with these fields:
        {{
            "century_probability": float,
            "reasoning": string,
            "confidence": string
        }}
        """
        result_text = None
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                temperature=0,
                stop=["\n\n"]
            )
            result_text = response['choices'][0]['text'].strip()
            # Match text within curly braces
            # Extract JSON object from response text
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            result = json.loads(match.group(0))
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {
                "century_probability": None,
                "reasoning": f"Error in processing: {str(e)}",
                "confidence": "low",
                "raw_response": result_text
            }

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataset of risk estimates.

        Parameters:
        df: Input DataFrame with risk estimates

        Returns:
        Processed DataFrame with standardized estimates
        """
        # Create copy to preserve original data
        processed_df = df.copy()

        # Process each row
        results = []
        processed_count = 0
        for idx, row in processed_df.iterrows():
            result = self.parse_estimate(row)
            results.append(result)
            processed_count += 1
            print(f"Processed row {processed_count} of {len(processed_df)}")

        # Add new columns
        processed_df['century_probability'] = [
            r['century_probability'] for r in results]
        processed_df['conversion_reasoning'] = [r['reasoning']
                                                for r in results]
        processed_df['conversion_confidence'] = [
            r['confidence'] for r in results]

        return processed_df

    def validate_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use LLM to validate and flag potential issues with converted estimates.

        Parameters:
        df: Processed DataFrame

        Returns:
        DataFrame with validation flags
        """
        def validate_row(row):
            prompt = f"""
            Task: Validate this existential risk estimate conversion:

            Original: {row['estimation']}
            Converted (per century): {row['century_probability']}
            Reasoning: {row['conversion_reasoning']}

            Are there any potential issues or inconsistencies?
            Output only "valid" or describe the specific issue briefly.
            """

            response = self.llm(
                prompt,
                max_tokens=64,
                temperature=0
            )

            return response['choices'][0]['text'].strip()

        df['validation_notes'] = df.apply(validate_row, axis=1)
        return df
