import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    """
    Returns:
      - vector of length 126 = (Left 63) + (Right 63)
      - annotated frame
      - debug info dict
    """
    def __init__(self, max_num_hands=2, detection_conf=0.6, track_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )

    def _landmarks_to_vec63(self, hand_landmarks):
        vec = []
        for lm in hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)  # (63,)

    def process(self, frame_bgr, draw=True):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_vec = np.zeros((63,), dtype=np.float32)
        right_vec = np.zeros((63,), dtype=np.float32)
        found_left = False
        found_right = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handed.classification[0].label  # "Left" or "Right"
                vec63 = self._landmarks_to_vec63(hand_lms)

                if label == "Left":
                    left_vec = vec63
                    found_left = True
                elif label == "Right":
                    right_vec = vec63
                    found_right = True

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )

        vec126 = np.concatenate([left_vec, right_vec], axis=0)  # (126,)

        info = {
            "found_left": found_left,
            "found_right": found_right,
        }
        return vec126, frame_bgr, info
