"""
System 21: Cable Health Monitoring — LSTM Anomaly Detection

Neural network-based cable health monitoring using LSTM for anomaly detection
and time-to-failure prediction. Detects incipient cable failure modes before
catastrophic breakage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CableHealthInput:
    """
    Time series input for cable health LSTM model.

    Attributes:
        tension_window: 1500x6 array of cable state time series.
            Dimension 1: 1500 time steps at 10 Hz = 150 seconds history
            Dimension 2: 6 features per timestep:
                [tension_n, tension_rate_n_s, spool_vel_m_s, rover_speed_m_s,
                 rover_yaw_rad_s, deployed_length_m]
            Used for learning temporal patterns of stress, motion, and failure.
    """

    tension_window: npt.NDArray[np.float32]  # shape (1500, 6)


@dataclass
class CableHealthOutput:
    """
    Cable health assessment and failure prediction output.

    Attributes:
        anomaly_score: [0, 1] probability of cable fault. 0 = healthy,
            1 = imminent failure.
        fault_class: Predicted failure mode string:
            - 'normal': Healthy cable operation
            - 'rock_wrap': Cable wrapped around rock causing local stress
            - 'slack_pile': Excessive slack causing tangling/friction loss
            - 'over_tension': Sustained tension above safe limits
            - 'burial': Cable buried under regolith
            - 'fatigue': Incipient metal fatigue from cyclic loading
        time_to_event_s: Estimated time until failure (seconds). Inf if healthy.
            Used to trigger preemptive recovery actions (slow rover, cut cable).
        confidence: [0, 1] model confidence in prediction. High confidence
            enables mission-critical decisions (abort, emergency deploy).
    """

    anomaly_score: float
    fault_class: str
    time_to_event_s: float
    confidence: float


class CableHealthMonitor(ABC):
    """
    LSTM-based cable health monitoring system.

    Uses 2-layer LSTM with hidden state 128 to detect cable faults before
    catastrophic failure. Monitors 150-second windows of cable state at 10 Hz,
    outputs anomaly score and failure mode classification.

    Model architecture:
    - Input: 1500 x 6 time series (150s at 10 Hz, 6 features)
    - LSTM layer 1: 128 hidden units
    - LSTM layer 2: 128 hidden units
    - 3-head output:
        Head 1: Anomaly score [0, 1] (sigmoid)
        Head 2: Fault class logits (5 classes: normal, rock_wrap, slack_pile,
                over_tension, burial, fatigue)
        Head 3: Time-to-failure regression [0, inf) seconds

    Inference at 2 Hz enables monitoring and decision-making on ~0.5s timescale
    while batching GPU computations.
    """

    @abstractmethod
    def initialize(self, model_path: str) -> None:
        """
        Load trained LSTM model from disk.

        Loads model weights and architecture from saved checkpoint. Should
        be called before any infer() calls.

        Args:
            model_path: Path to saved model file (.pth, .onnx, or similar).

        Returns:
            None

        Raises:
            FileNotFoundError: If model file not found.
            RuntimeError: If model format unsupported or load fails.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, input_data: CableHealthInput) -> CableHealthOutput:
        """
        Predict cable health from time series input.

        Runs LSTM forward pass on cable state time series and outputs
        health assessment. Called at 2 Hz during operation.

        Args:
            input_data: 1500x6 time series window of cable state.

        Returns:
            CableHealthOutput with anomaly score, fault class, and TTF estimate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_architecture(self) -> dict:
        """
        Get model architecture specification.

        Returns:
            Dict with architecture details:
            ```
            {
                'type': 'LSTM',
                'num_layers': 2,
                'hidden_size': 128,
                'input_dim': 6,
                'sequence_length': 1500,
                'output_heads': {
                    'anomaly_score': {'type': 'sigmoid', 'dim': 1},
                    'fault_class': {'type': 'softmax', 'dim': 5},
                    'time_to_event': {'type': 'relu', 'dim': 1}
                },
                'inference_rate_hz': 2
            }
            ```
        """
        raise NotImplementedError
