"""Example: Optimize a TFLite model for STM32MP1 using ST Edge AI DC.

This script demonstrates the end-to-end flow:
    1. Authenticate with ST Edge AI Developer Cloud
    2. Upload a .tflite model
    3. Generate NBG (Neural Binary Graph) for STM32MP1
    4. Download the optimized output

Prerequisites:
    - stm32ai_dc installed:
        pip install stm32ai_dc@git+https://github.com/STMicroelectronics/
        stm32ai-modelzoo-services.git#subdirectory=common/stm32ai_dc
    - Environment variables set:
        export STEDGE_USERNAME="your_email@example.com"
        export STEDGE_PASSWORD="your_password"

Usage:
    python examples/optimize_for_mpu.py --model build/best.tflite
    python examples/optimize_for_mpu.py --model build/best.tflite --timeout 900
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.stedge_wrapper import STEdgeAIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize a TFLite model for STM32MP1 via ST Edge AI DC",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .tflite model file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for NBG generation in seconds (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        default="build/mpu_output",
        help="Output directory for downloaded files (default: build/mpu_output)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only upload and analyze, skip NBG generation",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        return 1

    # Resolve model type from extension
    model_type = model_path.suffix.lower().lstrip(".")
    if model_type not in ("tflite", "onnx", "h5"):
        logger.error("Unsupported model type: %s (expected .tflite, .onnx, or .h5)", model_type)
        return 1

    try:
        # Step 1: Connect
        logger.info("Connecting to ST Edge AI Developer Cloud...")
        client = STEdgeAIClient()

        # Step 2: Upload
        logger.info("Uploading model: %s", model_path.name)
        client.upload_model(model_path)

        # Step 3: Analyze (optional)
        logger.info("Analyzing model...")
        analysis = client.analyze(model_path.name, model_type=model_type)
        logger.info("  ROM size:    %d bytes", analysis.rom_size)
        logger.info("  RAM size:    %d bytes", analysis.ram_size)
        logger.info("  MACC:        %d", analysis.macc)

        if args.analyze_only:
            logger.info("Analysis complete. Skipping NBG generation.")
            return 0

        # Step 4: Generate NBG
        logger.info("Generating NBG (timeout=%ds)...", args.timeout)
        blob_name = client.generate_nbg(model_path.name, timeout=args.timeout)
        logger.info("  NBG blob: %s", blob_name)

        # Step 5: Download
        output_dir = Path(args.output_dir)
        logger.info("Downloading to: %s", output_dir)
        nb_path = client.download_model(blob_name, output_dir)
        logger.info("  Downloaded: %s", nb_path)

        logger.info("Done! NBG file ready for deployment.")
        return 0

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
