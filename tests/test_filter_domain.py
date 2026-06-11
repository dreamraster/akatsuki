import unittest
import json
import requests
from unittest.mock import patch, MagicMock
from filter_domain import URLManager, batch_check_domain_match, process_batch

class TestFilterDomain(unittest.TestCase):
    def test_url_manager_rotation(self):
        """Test that URLManager correctly rotates through available endpoints."""
        urls = ["http://instance1:1234/v1", "http://instance2:1234/v1"]
        manager = URLManager(urls)
        
        self.assertEqual(manager.get_next_url(), "http://instance1:1234/v1")
        self.assertEqual(manager.get_next_url(), "http://instance2:1234/v1")
        self.assertEqual(manager.get_next_url(), "http://instance1:1234/v1")

    @patch("requests.post")
    def test_batch_check_domain_match_parsing(self, mock_post):
        """Test that the batched LLM response is correctly parsed into individual results."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        "Item 1: Reasoning: Explains math logic, Match: Yes\n"
                        "Item 2: Reasoning: Discusses weather, Match: No\n"
                        "Item 3: Reasoning: Simple arithmetic, Match: Yes"
                    )
                }
            }]
        }
        mock_post.return_value = mock_response

        texts = ["1+1=2", "It is raining", "5*5=25"]
        results, raw = batch_check_domain_match(texts, "math", "http://localhost", "local-model")

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0][0])
        self.assertIn("math logic", results[0][1])
        
        self.assertFalse(results[1][0])
        self.assertIn("weather", results[1][1])
        
        self.assertTrue(results[2][0])
        self.assertIn("arithmetic", results[2][1])

    @patch("requests.post")
    def test_batch_check_domain_match_connection_error(self, mock_post):
        """Test behavior when the LLM server is unreachable."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        texts = ["text1", "text2"]
        results, raw = batch_check_domain_match(texts, "domain", "http://localhost", "model")
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r[0] is False for r in results))
        self.assertIn("Failed to communicate", results[0][1])

    def test_process_batch_integration(self):
        """Test the process_batch worker function with mocked networking."""
        url_manager = URLManager(["http://mock-url"])
        batch_data = [
            {'text_to_eval': 'text 1', 'original_line': '{"line": 1}'},
            {'text_to_eval': 'text 2', 'original_line': '{"line": 2}'}
        ]
        
        with patch('filter_domain.batch_check_domain_match') as mock_check:
            mock_check.return_value = ([(True, "reason 1"), (False, "reason 2")], "raw response")
            
            results = process_batch(batch_data, "domain", url_manager, "model")
            
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0]['is_match'])
            self.assertEqual(results[0]['reasoning'], "reason 1")
            self.assertFalse(results[1]['is_match'])
            self.assertEqual(results[1]['reasoning'], "reason 2")
            self.assertEqual(results[0]['raw_response'], "raw response")

    @patch("requests.post")
    def test_batch_parsing_with_noise(self, mock_post):
        """Test parsing when the LLM adds extra conversational noise to the response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        "Sure, I can help you with that! Here are the evaluations:\n\n"
                        "Item 1: Reasoning: High quality logic, Match: Yes\n"
                        "Let me look at the second one...\n"
                        "Item 2: Reasoning: Just a greeting, Match: No\n\n"
                        "I hope this helps!"
                    )
                }
            }]
        }
        mock_post.return_value = mock_response

        texts = ["Logic test", "Hello there"]
        results, _ = batch_check_domain_match(texts, "general", "http://localhost", "model")

        self.assertTrue(results[0][0])
        self.assertFalse(results[1][0])

if __name__ == "__main__":
    unittest.main()
