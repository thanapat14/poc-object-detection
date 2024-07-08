# PoC Object Detection

This project demonstrates a Proof-of-Concept (PoC) for object detection using gRPC communication. It utilizes pre-trained models (e.g., `yolov8n.pt` and `yolov5s.pt`) to detect objects in images.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project provides a PoC for object detection using gRPC. It leverages pre-trained models (`yolov8n.pt` and `yolov5s.pt`) to perform object detection on images. The implementation includes both server and client components to demonstrate the functionality.

## Requirements

- Python 3.10 (with Poetry installed)
- gRPC tools (`protoc`)

## Installation

1. Create a Poetry shell:
    ```sh
    poetry shell
    ```

2. Install the dependencies using Poetry:
    ```sh
    poetry install
    ```

3. Compile the gRPC protobuf file:
    ```sh
    poetry run python -m grpc_tools.protoc -I. --python_out=src/poc/object_detection/ --grpc_python_out=src/poc/object_detection/ protos/object_detection.proto
    ```

4. Run the gRPC server:
    ```sh
    poetry run python src/poc/object_detection/server.py
    ```

## Usage

To run the client and perform object detection on an image, use the following commands:

```sh
poetry run python src/poc/object_detection/client.py --image_filepath "resources/image_1.jpg"
```

To specify a different model, use the --model_name parameter:

```sh
poetry run python src/poc/object_detection/client.py --image_filepath "resources/image_2.jpg" --model_name "yolov8n.pt"
poetry run python src/poc/object_detection/client.py --image_filepath "resources/image_3.jpg" --model_name "yolov5s.pt"
poetry run python src/poc/object_detection/client.py --image_filepath "resources/image_4_not_detect.jpg" --model_name "yolov5s.pt"
```
