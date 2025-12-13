import argparse
from settings import settings
from src.scripts.download_model_artifacts import download_artifacts
from src.scripts.export_sentence_transformer_to_onnx import export_model_to_onnx
from src.scripts.export_classifier_to_onnx import export_classifier_to_onnx


def main():
    arg_parser = argparse.ArgumentParser(
        description="Download and export model to ONNX"
    )
    arg_parser.add_argument(
        "--script",
        type=str,
        choices=["download", "export"],
        required=True,
    )

    args = arg_parser.parse_args()

    match args.script:
        case "download":
            download_artifacts(settings)
        case "export":
            export_model_to_onnx(settings)
            export_classifier_to_onnx(settings)


if __name__ == "__main__":
    main()