# import: standard
import argparse
import logging
import sys
import pathlib
from typing import List

# import: internal
from poc.object_detection.protos import object_detection_pb2
from poc.object_detection.protos import object_detection_pb2_grpc
from poc.utils.read_file import is_image_file

# import: external
import grpc


def args_handler(argv: List[str] = sys.argv[1:]) -> argparse.Namespace:
    """
    Handle command line arguments for the script.

    Args:
        argv (List[str]): List of command line arguments. Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Entrypoint for object detection demo")
    parser.add_argument(
        "--image_filepath", help="Image file path for prediction", required=True, type=str
    )
    parser.add_argument(
        "--model_name", help="Model name to use for prediction", required=False, type=str
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity level for logging (INFO or DEBUG, default: INFO)",
        required=False,
        choices=["INFO", "DEBUG"],
        default="INFO",
    )

    # Parse command line arguments using the defined argument parser
    system_arguments = parser.parse_args(argv)

    # Return the parsed and processed command line arguments
    return system_arguments


def main():
    """Main Function of Application"""
    # Parse command line arguments using the defined argument handler function
    system_arguments = args_handler()

    # Convert verbosity argument to corresponding logging level
    if system_arguments.verbosity == "INFO":
        logging_level = logging.INFO
    elif system_arguments.verbosity == "DEBUG":
        logging_level = logging.DEBUG

    # Configure logging
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s][%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ObjectDetectionClient")

    logger.info("Starting the object detection client...")

    image_filepath = system_arguments.image_filepath
    logger.info(f"image_filepath={image_filepath}")

    model_name = system_arguments.model_name
    logger.info(f"model_name={model_name}")

    # Check if the specified file path is a valid image file
    logger.info("Checking if the provided file path is a valid image file...")
    if not is_image_file(image_filepath):
        logger.error(f"'{image_filepath}' is not a valid image file.")
        raise ValueError(f"'{image_filepath}' is not a valid image file.")
    logger.info("Image file is valid.")

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = object_detection_pb2_grpc.ObjectDetectionStub(channel)

        # Read the image file as bytes object
        logger.info(f"Reading the image file: {image_filepath}")
        image_data = pathlib.Path(image_filepath).read_bytes()

        # Create a DetectionRequest with the image data, image file path and model name
        request = object_detection_pb2.DetectionRequest(
            image=image_data,
            image_file_path=image_filepath,
            model_name=model_name,
        )

        logger.info("Sending detection request to the gRPC server...")
        try:
            response = stub.Detect(request)
        except grpc.RpcError as e:
            logger.error(f"Error during RPC call: {e}")
            return

        logger.info("Received detection response from the gRPC server.")
        # Log a warning if no objects were detected
        if not response.detections:
            logger.warning("No objects detected.")
            return None

        # Log individual detection details
        logger.debug(f"response: {response}")
        for detection in response.detections:
            logger.info(
                f"Label: {detection.label}, Confidence: {detection.confidence}, "
                f"BBox: ({detection.x}, {detection.y}, {detection.width}, {detection.height})"
            )

        # Return the result as a string (for simplicity, you can customize this)
        result_str = f"Detected objects: {response}"
        return result_str


if __name__ == "__main__":
    main()
