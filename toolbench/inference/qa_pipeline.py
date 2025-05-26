#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Close-domain QA Pipeline

A tool-enhanced question answering system based on large language models, 
supporting multiple inference methods and tool invocations.
"""

import argparse
import logging
import os
import json
from typing import Dict, Any
from utils import CONFIG_DESCRIPTION


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Close-domain QA Pipeline")

    # Configuration file
    parser.add_argument('--config_file', type=str, default='config/default.json',
                        help='Path to configuration file')

    # API Configuration
    parser.add_argument('--base_url', type=str,
                        help=CONFIG_DESCRIPTION["base_url"])
    parser.add_argument('--openai_key', type=str,
                        help=CONFIG_DESCRIPTION["openai_key"])

    # Model Configuration
    parser.add_argument('--backbone_model', type=str,
                        help=CONFIG_DESCRIPTION["backbone_model"])
    parser.add_argument('--chatgpt_model', type=str,
                        help=CONFIG_DESCRIPTION["chatgpt_model"])
    parser.add_argument('--model_path', type=str,
                        help=CONFIG_DESCRIPTION["model_path"])
    parser.add_argument("--lora", action="store_true",
                        help=CONFIG_DESCRIPTION["lora"])
    parser.add_argument('--lora_path', type=str,
                        help=CONFIG_DESCRIPTION["lora_path"])

    # Inference Configuration
    parser.add_argument('--max_observation_length', type=int,
                        help=CONFIG_DESCRIPTION["max_observation_length"])
    parser.add_argument('--max_source_sequence_length', type=int,
                        help=CONFIG_DESCRIPTION["max_source_sequence_length"])
    parser.add_argument('--max_sequence_length', type=int,
                        help=CONFIG_DESCRIPTION["max_sequence_length"])
    parser.add_argument('--observ_compress_method', type=str,
                        choices=["truncate", "filter", "random"],
                        help=CONFIG_DESCRIPTION["observ_compress_method"])
    parser.add_argument('--method', type=str,
                        help=CONFIG_DESCRIPTION["method"])

    # Data Configuration
    parser.add_argument('--input_query_file', type=str,
                        help=CONFIG_DESCRIPTION["input_query_file"])
    parser.add_argument('--input_query_dir', type=str,
                        help=CONFIG_DESCRIPTION["input_query_dir"])
    parser.add_argument('--output_answer_file', type=str,
                        help=CONFIG_DESCRIPTION["output_answer_file"])
    parser.add_argument('--tool_root_dir', type=str,
                        help=CONFIG_DESCRIPTION["tool_root_dir"])
    parser.add_argument('--gt_data_file', type=str,
                        help=CONFIG_DESCRIPTION["gt_data_file"])

    # Service Configuration
    parser.add_argument('--toolbench_key', type=str,
                        help=CONFIG_DESCRIPTION["toolbench_key"])
    parser.add_argument('--rapidapi_key', type=str,
                        help=CONFIG_DESCRIPTION["rapidapi_key"])
    parser.add_argument('--use_rapidapi_key', action="store_true",
                        help=CONFIG_DESCRIPTION["use_rapidapi_key"])
    parser.add_argument('--api_customization', action="store_true",
                        help=CONFIG_DESCRIPTION["api_customization"])

    # Other Configuration
    parser.add_argument('--device', type=str,
                        help=CONFIG_DESCRIPTION["device"])
    parser.add_argument('--attack', type=str,
                        help=CONFIG_DESCRIPTION["attack"])
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """
    Load configuration from file and override with command-line arguments

    Args:
        args: Command-line arguments

    Returns:
        Configuration object
    """
    config = {}

    # Load from config file
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
            logging.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logging.error(f"Failed to load configuration file: {str(e)}")
            raise

    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config_file':
            config[key] = value

    # Create a Config object to allow attribute-style access
    return Config(**config)


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Configure logging

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters

    Args:
        config: Configuration object
    """
    required_fields = [
        'base_url', 'openai_key', 'chatgpt_model',
        'model_path', 'input_query_file', 'output_answer_file'
    ]

    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required configuration field: {field}")

    # Additional validation logic can be added here


def main() -> None:
    """Main function"""
    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        return

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        return

    # Set environment variables
    os.environ['SERVICE_URL'] = getattr(config, 'service_url', "http://localhost:8080/virtual")
    os.environ['CUDA_VISIBLE_DEVICES'] = getattr(config, 'cuda_device', '0')

    # Import pipeline runner
    try:
        from toolbench.inference.Downstream_tasks.rapidapi import pipeline_runner
        logger.info("Pipeline runner imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import pipeline runner: {str(e)}")
        return

    logger.info(f"Attack type: {getattr(config, 'attack', None)}")

    # Initialize and run the pipeline
    try:
        logger.info("Initializing close-domain QA pipeline...")
        runner = pipeline_runner(config)

        logger.info("Starting pipeline execution...")
        runner.run()

        logger.info("Pipeline execution completed successfully")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()