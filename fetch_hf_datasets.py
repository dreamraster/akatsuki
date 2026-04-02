# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""
Fetch datasets from Hugging Face.

Usage:
    pip install requests huggingface_hub
    python fetch_hf_datasets.py [org/username] [name_prefix]

Examples:
    # List all datasets from TeichAI
    python fetch_hf_datasets.py

    # List all datasets from a specific organization
    python fetch_hf_datasets.py TeichAI

    # List datasets starting with "coco" from an org
    python fetch_hf_datasets.py TeichAI coco

    # List datasets from another user/organization
    python fetch_hf_datasets.py HuggingFace Inc.

"""

import argparse
import os
import subprocess
import sys


def fetch_datasets(org_name, prefix=""):
    """
    Fetch datasets from a Hugging Face organization/user.

    Args:
        org_name: The username or organization name
        prefix: Optional filter for dataset names

    Returns:
        list: List of dataset information dicts with download URLs
    """
    # Try to import huggingface_hub
    try:
        from huggingface_hub import list_datasets
    except ImportError:
        print("huggingface-hub not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import list_datasets

    try:
        print("-" * 70)
        print(f"Fetching datasets from: {org_name}")
        if prefix:
            print(f"Filtering: Datasets starting with '{prefix}'")
        print("-" * 70)

        # List all datasets for the organization/user
        datasets_list = list(list_datasets(author=org_name))

        if not datasets_list:
            print("No datasets found.")
            return []

        # Filter by prefix if provided
        if prefix:
            filtered_datasets = [
                d for d in datasets_list if prefix.lower() in d.id.lower()
            ]
            print(f"Found {len(filtered_datasets)} out of {len(datasets_list)} datasets\n")
        else:
            filtered_datasets = datasets_list

        results = []
        for i, dataset in enumerate(filtered_datasets, 1):
            dataset_id = dataset.id
            # Construct the direct Hugging Face dataset URL
            hf_page_url = f"https://huggingface.co/datasets/{dataset_id}"

            results.append({
                "name": dataset_id,
                "hf_link": hf_page_url,
            })

            print(f"[{i}] {dataset_id}")
            print(f"    HuggingFace Page: {hf_page_url}")

            # Show curl command to get direct access
            print("    Usage with curl:")
            print(f"        hf download {dataset_id} --repo-type dataset")
            print()

        if not filtered_datasets:
            print(f"No datasets found matching prefix '{prefix}'")

        print("=" * 70)
        print(f"Total: {len(results)} dataset(s) found")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"Error fetching datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and optionally download Hugging Face datasets"
    )
    parser.add_argument("org", nargs="?", const="TeichAI", default="TeichAI",
                        help="Organization or username (default: TeichAI)")
    parser.add_argument("prefix", nargs="?",
                         help="Filter datasets starting with this prefix")

    args = parser.parse_args()

    org_name = args.org
    name_prefix = args.prefix

    print("=" * 70)
    if name_prefix:
        print(f"Searching for datasets starting with: '{name_prefix}'")
    print(f"Organization: {org_name}")
    print()

    datasets = fetch_datasets(org_name, name_prefix)

    if datasets is not None:
        print()
        if not datasets:
            print("No datasets found.")
            return 1
        else:
            print(f"Total: {len(datasets)} dataset(s) found")
        print("=" * 70)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
