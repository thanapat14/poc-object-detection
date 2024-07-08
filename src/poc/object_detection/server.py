# import: standard
import io
import logging
import os
from concurrent import futures
from typing import List

# import: internal
from poc.object_detection.protos import object_detection_pb2
from poc.object_detection.protos import object_detection_pb2_grpc

# import: external
import cv2
import grpc
from PIL import Image
from ultralytics import YOLO


class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServicer):
    def __init__(self):
        self.logger = logging.getLogger("ObjectDetectionServer")

    def select_yolo_model(
        self,
        model_name: str = "yolov8n.pt",
        task: str = "detect",
        verbose: bool = False,
    ) -> YOLO:
        """
        Selects and returns the YOLO model based on the specified model name.

        Args:
            model_name (str, optional): Name of the YOLO model to select. Defaults to "yolov8n.pt".

        Returns:
            YOLO: YOLO model instance based on the selected model name.
        """
        self.logger.info(f"Model name: {model_name}")
        self.logger.info(f"Task: {task}")
        self.logger.debug(f"Verbose: {verbose}")
        return YOLO(model=model_name, task=task, verbose=verbose)

    def prepare_detections(self, results: List) -> List[object_detection_pb2.Detection]:
        """
        Prepare detection objects from YOLO results.

        Args:
            results (List): YOLO detection results.

        Returns:
            List[object_detection_pb2.Detection]: List of Detection objects.
        """
        detections: List[object_detection_pb2.Detection] = []
        for result in results:
            for bbox in result.boxes:
                detection = object_detection_pb2.Detection(
                    label=str(int(bbox.cls.item())),
                    confidence=float(bbox.conf[0].item()),
                    x=float(bbox.xywh[0][0].item()),
                    y=float(bbox.xywh[0][1].item()),
                    width=float(bbox.xywh[0][2].item()),
                    height=float(bbox.xywh[0][3].item()),
                )
                detections.append(detection)
        return detections

    def annotate_image(
        self,
        image_file_path: str,
        results: List,
    ) -> str:
        """
        Annotate the input image with detection results.

        Args:
            image_file_path (str): File path of the input image.
            results (List): YOLO detection results.

        Returns:
            str: File path of the annotated image.
        """
        file_name, file_extension = os.path.splitext(image_file_path)
        annotated_image_path = file_name + "_annotated" + file_extension

        for result in results:
            annotated_image = result.plot()  # Assuming plot() returns the annotated image
            cv2.imwrite(annotated_image_path, annotated_image)

        return annotated_image_path

    def Detect(
        self,
        request: object_detection_pb2.DetectionRequest,
        context: grpc.ServicerContext,
    ) -> object_detection_pb2.DetectionResponse:
        """
        Perform object detection on the provided image data.

        Args:
            request (object_detection_pb2.DetectionRequest): Request containing image data and model name.
            context (grpc.ServicerContext): Context object for the RPC call.

        Returns:
            object_detection_pb2.DetectionResponse: Response containing detections.
        """
        image = Image.open(io.BytesIO(request.image))

        if request.model_name:
            model = self.select_yolo_model(model_name=request.model_name)
        else:
            model = self.select_yolo_model()

        try:
            self.logger.info("Running object detection model.")
            results = model(image)
        except Exception as e:
            self.logger.error(f"Error during object detection: {e}")
            context.set_details(details=f"Error during object detection: {e}")
            context.set_code(code=grpc.StatusCode.INTERNAL)
            return object_detection_pb2.DetectionResponse(detections=[])

        # Check if no results were returned
        for result in results:
            if len(result.boxes) == 0:
                self.logger.info("No objects detected.")
                return object_detection_pb2.DetectionResponse(detections=[])

        # Prepare the detection response
        detections = self.prepare_detections(results)

        # Annotate the image with the detection results
        annotated_image_path = self.annotate_image(request.image_file_path, results)
        self.logger.info(f"Annotated image saved to: {annotated_image_path}")

        self.logger.info("Detection completed successfully.")
        return object_detection_pb2.DetectionResponse(detections=detections)


def serve() -> None:
    """
    Start the gRPC server for object detection and wait for incoming requests.

    This function initializes the gRPC server, binds it to a port, starts it,
    and waits for incoming requests indefinitely.
    """
    # Configure logging settings
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s][%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ObjectDetectionServer")

    logger.info("Starting the gRPC server.")

    # Create a gRPC server instance with thread pool executor
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=10))

    # Add ObjectDetectionServicer to the server
    object_detection_pb2_grpc.add_ObjectDetectionServicer_to_server(
        servicer=ObjectDetectionServicer(),
        server=server,
    )

    # Bind the server to an insecure port and start the server
    server.add_insecure_port(address="[::]:50051")
    server.start()

    logger.info("Server is running ...")  # Log server start confirmation

    # Wait for the server to terminate (blocks indefinitely)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
