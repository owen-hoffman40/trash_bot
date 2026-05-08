# SOURCE REFERENCES USED IN THIS FILE:
# - PyTorch Module API:
#   https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# - PyTorch inference performance checklist:
#   https://docs.pytorch.org/serve/performance_checklist.html
# - torchvision ResNet18:
#   https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
# - torchvision transforms:
#   https://pytorch.org/vision/stable/transforms.html
# - ROS sensor_msgs/Image message layout:
#   https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html
# - ROS image encoding constants:
#   https://docs.ros.org/en/rolling/p/sensor_msgs/generated/program_listing_file_include_sensor_msgs_image_encodings.hpp.html
# - ROS 2 Python publisher/subscriber tutorial:
#   https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
# - OpenCV:
#   https://opencv.org
# - NumPy:
#   https://numpy.org
# - Pillow Image module:
#   https://pillow.readthedocs.io/en/stable/reference/Image.html

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from torchvision import models, transforms


MODEL_PATH = Path("/home/pi/E90_ws/src/CV/trashbot_dir/classifier.pt")
# For RGB imaging
IMAGE_TOPIC = "/camera/camera/color/image_raw"
# Label encodings specific to this classifier
LABELS = ["Compost", "Recycling", "Trash"]

# Params to alter for best performance
CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_LABEL = "Trash"

BUFFER_SECONDS = 0.5
SAMPLE_COUNT = 5
FRAME_STRIDE = 8
FRAME_SAMPLE_SLEEP_SECONDS = 0.03

SAVED_IMAGE_OUTPUT_DIR = Path("/home/pi/E90_ws/src/CV/trashbot_dir/classified_frame_outputs")


def ros_image_to_bgr(img_msg):
    # ROS gives us a flat byte buffer, so we manually reshape it into an image array
    dtype = np.uint8
    n_channels = 3

    img_buf = np.frombuffer(img_msg.data, dtype=dtype)

    image = img_buf.reshape(
        img_msg.height,
        int(img_msg.step / np.dtype(dtype).itemsize / n_channels),
        n_channels,
    )

    # ROS images can include padding, clip it to our img size to ensure its not there
    image = image[: img_msg.height, : img_msg.width, :]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class FramePrediction:
    # Per frame prediction class
    # label is the post-threshold label, while raw_label is the model's direct top guess
    def __init__(self, label, confidence, raw_label, raw_confidence):
        self.label = label
        self.confidence = confidence
        self.raw_label = raw_label
        self.raw_confidence = raw_confidence


class ClassificationResult:
    # label is the final voted label and predictions are the frame-wise predictions
    # Useful if you want to debug how the final answer was chosen
    def __init__(self, label, predictions):
        self.label = label
        self.predictions = predictions


class ResNet18VideoClassifier:
    # This class handles the ResNet18 model:
    # preprocess frames, run inference, optionally save sampled frames, and combine predictions
    def __init__(self, model, labels, confidence_threshold, low_confidence_label, device=None):
        # Check to see if GPU is available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.low_confidence_label = low_confidence_label

        # This matches the ImageNet-style preprocessing you'd usually see with torchvision models
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    # Preprocessing of the image
    def prepare_input(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(frame_rgb)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    def make_prediction(self, raw_label, raw_confidence):
        label = raw_label

        # Below is the confidence cut off
        if self.low_confidence_label is not None:
            if raw_confidence < self.confidence_threshold:
                label = self.low_confidence_label

        return FramePrediction(label, raw_confidence, raw_label, raw_confidence)

    # Run this function in PyTorch inference mode so it skips gradient tracking
    @torch.inference_mode()
    def classify_frame(self, frame_bgr):
        # Turn the OpenCV frame into the tensor format the model expects
        input_tensor = self.prepare_input(frame_bgr)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get the top predicted class and corresponding confidence score
        confidence_tensor, class_index_tensor = torch.max(probabilities, dim=1)
        class_index = int(class_index_tensor.item())
        raw_confidence = float(confidence_tensor.item())
        raw_label = self.labels[class_index]

        # Apply the confidence threshold before return
        return self.make_prediction(raw_label, raw_confidence)

    def save_classified_frame(self, frame_bgr, prediction, output_path):
        # Save individual images gathered from camera and put our model output on it for debugging
        output_image = frame_bgr.copy()

        cv2.putText(
            output_image,
            f"pred={prediction.label} raw={prediction.raw_label} conf={prediction.confidence:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

    # Take majority label
    def majority_vote(self, labels):
        counts = Counter(labels)
        return counts.most_common(1)[0][0]

    def classify_frames(self, frames, frame_stride, saved_image_output_dir=SAVED_IMAGE_OUTPUT_DIR):
        predictions = []
        labels = []
        frame_index = 0

        for frame in frames:
            frame_index += 1

            # Below is important to ensure that the images are taken with enough of a time gap,
            # so it is not 5 consecutive images
            if frame_index % frame_stride != 0:
                continue

            prediction = self.classify_frame(frame)

            # Save image, classificaton, and confidence for debugging
            sample_index = len(predictions) + 1
            output_path = saved_image_output_dir / f"classified_frame_{sample_index}.jpg"
            self.save_classified_frame(frame, prediction, output_path)

            predictions.append(prediction)
            labels.append(prediction.label)

        final_label = self.majority_vote(labels)

        # Return object with the final label and preditcions for each of the "votes"
        return ClassificationResult(final_label, predictions)


def load_resnet18_model(model_path, class_count, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # The way the model is saved is learned weights only, so first we must rebuild the architecture
    model_ft = models.resnet18(weights=None)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, class_count)

    # Now we load the weights in
    model_ft.load_state_dict(checkpoint["model_state"])
    model_ft.to(device)
    model_ft.eval()

    return model_ft


class RosFrameBufferNode(Node):
    # This class ensures easy access to the most recent frame
    # so the classifier can grab it whenever it needs one.
    def __init__(self, image_topic):
        super().__init__("ros_classify_frame_buffer")

        self.lock = threading.Lock()
        self.latest_frame = None

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

    def image_callback(self, msg):
        frame = ros_image_to_bgr(msg)

        # The ROS callback writes frames while the classifier reads them,
        # Which is a race condition
        # The lock makes sure only one thread accesses latest_frame at a time
        with self.lock:
            self.latest_frame = frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            # Return a copy to ensure that the updating frames do not impact the classifier
            return self.latest_frame.copy()


class RosVideoItemClassifier:
    # A nice high level wrapper to put everything together
    # It starts the ROS subscriber, collects a short burst of frames, and returns one final label.
    def __init__(
        self,
        model_path=MODEL_PATH,
        image_topic=IMAGE_TOPIC,
        labels=LABELS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        low_confidence_label=LOW_CONFIDENCE_LABEL,
    ):
        if not rclpy.ok():
            rclpy.init()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_resnet18_model(
            model_path=model_path,
            class_count=len(labels),
            device=self.device,
        )

        self.classifier = ResNet18VideoClassifier(
            model=self.model,
            labels=labels,
            confidence_threshold=confidence_threshold,
            low_confidence_label=low_confidence_label,
            device=self.device,
        )

        self.executor = SingleThreadedExecutor()
        self.node = RosFrameBufferNode(image_topic=image_topic)
        self.executor.add_node(self.node)

        # ROS spinning needs to stay alive while we do our own logic,
        # so we push it into a background thread instead of blocking the whole program.
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.started = False

    def start(self):
        if self.started:
            return

        self.spin_thread.start()
        self.started = True

    def stop(self):
        if not self.started:
            return

        self.executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        self.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        self.started = False

    def classify_once(
        self,
        buffer_seconds=BUFFER_SECONDS,
        sample_count=SAMPLE_COUNT,
        frame_stride=FRAME_STRIDE,
        saved_image_output_dir=SAVED_IMAGE_OUTPUT_DIR,
    ):
        while self.node.get_latest_frame() is None:
            time.sleep(0.05)

        # We wait a bit before collecting frames so we're classifying a short "window" of video
        # instead of classifying the same image a couple times before it is updated
        time.sleep(buffer_seconds)

        frames = []
        target_frame_count = sample_count * frame_stride

        while len(frames) < target_frame_count:
            frame = self.node.get_latest_frame()

            if frame is not None:
                frames.append(frame)

            # Sleep so we don't just grab the exact same frame object as fast as possible.
            time.sleep(FRAME_SAMPLE_SLEEP_SECONDS)

        result = self.classifier.classify_frames(
            frames,
            frame_stride,
            saved_image_output_dir,
        )

        return result.label


def classify_item(saved_image_output_dir=SAVED_IMAGE_OUTPUT_DIR):
    classifier = RosVideoItemClassifier()
    classifier.start()

    try:
        return classifier.classify_once(
            saved_image_output_dir=saved_image_output_dir,
        )
    finally:
        classifier.stop()


if __name__ == "__main__":
    print(classify_item())