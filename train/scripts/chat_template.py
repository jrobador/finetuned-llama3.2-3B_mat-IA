
#%%
import json
from typing import Dict, List, Optional, Union
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ConversationFormat:
    """Configuration for conversation format tokens."""
    begin_text: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    end_of_text: str = "<|eot_id|>"

class DatasetProcessor:
    """Process and convert datasets into a unified conversation format."""
    
    def __init__(
        self,
        format_config: Optional[ConversationFormat] = None,
    ):
        """
        Initialize the dataset processor.
        
        Args:
            format_config: Custom conversation format configuration
            output_dir: Directory to save processed datasets
        """
        self.format = format_config or ConversationFormat()
    
    def create_prompt(
        self,
        row: Dict,
        instruction: Optional[str] = None,
        input_key: str = "input",
        output_key: str = "output"
    ) -> str:
        """
        Create a formatted prompt from a data row.
        
        Args:
            row: Dictionary containing the data
            instruction: Optional system instruction
            input_key: Key for input text in the row
            output_key: Key for output text in the row
        
        Returns:
            Formatted prompt string
        """
        f = self.format
        parts = []
        
        parts.append(f"{f.begin_text}")
        
        if instruction:
            parts.extend([
                f"{f.start_header}system{f.end_header}",
                f"{instruction}",
                f"{f.end_of_text}"
            ])
        
        parts.extend([
            f"{f.start_header}user{f.end_header}",
            f"{row[input_key]}",
            f"{f.end_of_text}",
            f"{f.start_header}assistant{f.end_header}",
            f"{row[output_key]}",
            f"{f.end_of_text}"
        ])
        
        return " ".join(parts)
    
    def process_json_dataset(self, json_path: Union[str, Path]) -> Dataset:
        """
        Process a JSON dataset file.
        
        Args:
            json_path: Path to JSON dataset file
        
        Returns:
            Processed Hugging Face dataset
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompts = []
            for row in data:
                prompt = self.create_prompt(row)
                prompts.append({"prompt": prompt})
            
            return Dataset.from_pandas(pd.DataFrame(prompts))
        
        except Exception as e:
            raise RuntimeError(f"Error processing JSON dataset: {e}")
    
    def process_hf_datasets(
        self,
        dataset_configs: List[Dict],
        remove_ids: Optional[List[int]] = None
    ) -> Dataset:
        """
        Process multiple Hugging Face datasets.
        
        Args:
            dataset_configs: List of dataset configurations
            remove_ids: Optional list of IDs to remove
        
        Returns:
            Combined processed dataset
        """
        processed_datasets = []
        
        for config in dataset_configs:
            try:
                ds = load_dataset(
                    config["name"],
                    split=config.get("split", "train")
                )
                
                if remove_ids:
                    ds = ds.filter(
                        lambda _, idx: idx not in remove_ids,
                        with_indices=True
                    )
                
                if config.get("rename_columns"):
                    ds = ds.rename_columns(config["rename_columns"])
                
                ds = ds.map(
                    lambda x: {
                        "prompt": self.create_prompt(
                            x,
                            input_key=config["input_key"],
                            output_key=config["output_key"]
                        )
                    },
                    remove_columns=config.get("remove_columns", [])
                )
                
                processed_datasets.append(ds)
                
            except Exception as e:
                print(f"Error processing dataset {config['name']}: {e}")
                continue
        
        return concatenate_datasets(processed_datasets)

#%%
if __name__ == "__main__":
    processor = DatasetProcessor()
    
    dataset_configs = [
        {
            "name": "jrobador/mathinstruct_es",
            "input_key": "question",
            "output_key": "answer",
            "rename_columns": {"instruction": "question", "output": "answer"},
            "remove_columns": ["question", "answer", "Unnamed: 0", "source"]
        },
        {
            "name": "jrobador/gsm8k_es",
            "input_key": "question",
            "output_key": "answer",
            "remove_columns": ["question", "answer", "Unnamed: 0"]
        }
    ]
    
    # Process datasets
    remove_ids = [18865, 143408]  # IDs of questions without answers

    combined_dataset = processor.process_hf_datasets(dataset_configs, remove_ids)
