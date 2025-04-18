from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
import clip
from torchvision import transforms

from kts.cpd_auto import cpd_auto

class FeatureExtractor(object):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load('ViT-B/32', device=self.device)
        self.model = self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            preprocess
        ])

        self.linear = nn.Linear(self.model.visual.output_dim, 1024).to(self.device)
        if self.device.startswith("cuda"):
            self.linear = self.linear.half()  # Ensure linear layer matches CLIP's dtype

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.transform(img)
        batch = img.unsqueeze(0).to(self.device)

        if self.device.startswith("cuda"):
            batch = batch.half()  # Convert input to half precision

        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = self.linear(features)  # No need to cast again, both are in half precision
            features = features.squeeze().float().cpu().numpy()  # Convert back to float32 for processing

        assert features.shape == (1024,), f'Invalid feature shape {features.shape}: expected 1024'
        features /= linalg.norm(features) + 1e-10
        return features

class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)

            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks