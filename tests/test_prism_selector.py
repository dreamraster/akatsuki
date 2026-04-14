import unittest
import torch
import numpy as np
import os
import sys

# Add project root to path so we can import hmlcore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from unittest.mock import MagicMock
from hmlcore.prism_selector import select_with_prism, _extract_embeddings

class TestPrismSelector(unittest.TestCase):
    def setUp(self):
        # Create a toy dataset
        self.dataset = Dataset.from_list([
            {"prompt": "hello world", "completion": "hi"},
            {"prompt": "how are you", "completion": "good"},
            {"prompt": "what is 2+2", "completion": "4"},
            {"prompt": "tell me a joke", "completion": "funny"},
            {"prompt": "hello word", "completion": "hi"}, # redundant with 1st
            {"prompt": "2+2 equals what", "completion": "4"}, # redundant with 3rd
        ])
        
    def test_recentering_and_rowsums(self):
        # Mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Create a more robust synthetic dataset:
        # 40 redundant vectors (clustered)
        # 20 diverse vectors (orthogonal)
        torch.manual_seed(42)
        dim = 128
        
        # Cluster: 40 vectors pointing roughly towards [1, 0, ...]
        clustered = torch.randn(40, dim) * 0.01 + torch.tensor([1.0] + [0.0]*(dim-1))
        # Diverse: 20 vectors pointing towards different axes
        diverse = torch.zeros(20, dim)
        for i in range(20):
            diverse[i, i+1] = 1.0 # use different dims, starting from index 1 to avoid the cluster axis
            diverse[i] += torch.randn(dim) * 0.01
            
        # Normalize all to unit length (LLMs/VLMs usually have stable hidden state norms)
        mock_embeddings = torch.cat([clustered, diverse], dim=0)
        mock_embeddings = torch.nn.functional.normalize(mock_embeddings, p=2, dim=1)
        
        import hmlcore.prism_selector
        hmlcore.prism_selector._extract_embeddings = MagicMock(return_value=mock_embeddings)
        
        dataset = Dataset.from_list([{"prompt": f"p{i}"} for i in range(60)])
        
        # Test "high" tier (least redundant)
        filtered_ds = select_with_prism(dataset, model, tokenizer, tier="high")
        
        # With this setup, the 20 diverse outliers should have lower redundancy
        # since they are orthogonal to the 40-sample cluster.
        selected_indices = [int(p["prompt"][1:]) for p in filtered_ds]
        
        # Check count (roughly 1/3)
        self.assertAlmostEqual(len(selected_indices), 20, delta=2)
        
        # Check quality: the majority of selected samples should be from the diverse set (idx >= 40)
        diverse_count = sum(1 for idx in selected_indices if idx >= 40)
        self.assertGreater(diverse_count, 15, "PRISM failed to prioritize diverse outliers over the redundant cluster")
        
    def test_tier_split_logic(self):
        # Verify quantiles sum to N
        # We'll mock raw embeddings and check logic
        mock_embeddings = torch.randn(100, 128)
        import hmlcore.prism_selector
        hmlcore.prism_selector._extract_embeddings = MagicMock(return_value=mock_embeddings)
        
        model, tokenizer = MagicMock(), MagicMock()
        
        ds_high = select_with_prism(Dataset.from_list([{"prompt": "x"}]*100), model, tokenizer, tier="high")
        ds_mid  = select_with_prism(Dataset.from_list([{"prompt": "x"}]*100), model, tokenizer, tier="mid")
        ds_low  = select_with_prism(Dataset.from_list([{"prompt": "x"}]*100), model, tokenizer, tier="low")
        
        # Tier sizes should be roughly 1/3 each
        self.assertGreater(len(ds_high), 25)
        self.assertGreater(len(ds_mid), 25)
        self.assertGreater(len(ds_low), 25)
        self.assertEqual(len(ds_high) + len(ds_mid) + len(ds_low), 100)

if __name__ == "__main__":
    unittest.main()
