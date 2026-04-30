"""Vision-based interface for tic-tac-toe game.

This module uses machine vision to detect user moves on a physical or displayed tic-tac-toe board
and communicates with tic_tac_logic to get the robot's response.

Usage:
    python vision_interface.py

The board layout expected by the camera:
    1 2 3
    4 5 6
    7 8 9

Dependencies:
    pip install opencv-python numpy
"""

from __future__ import annotations

import subprocess
import sys
import cv2
import numpy as np
from typing import Optional

# Color ranges in HSV for detecting marks
# X marks (user) - typically blue
X_LOWER_HSV = np.array([100, 50, 50])
X_UPPER_HSV = np.array([130, 255, 255])

# O marks (robot) - typically red
O_LOWER_HSV = np.array([0, 100, 100])
O_UPPER_HSV = np.array([10, 255, 255])

# Board and grid detection
MIN_CONTOUR_AREA = 500
GRID_DIVISIONS = 3


class BoardDetector:
    """Detects tic-tac-toe board and cell states from video frames."""

    def __init__(self):
        self.previous_board_state: list[int] = [0] * 9
        self.board_region: Optional[tuple] = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to HSV and apply blur."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        return blurred

    def find_board_region(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect the board region using edge detection and contour analysis.
        Returns (x, y, w, h) or None if board not found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        board_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > MIN_CONTOUR_AREA * 10:
                max_area = area
                board_contour = contour

        if board_contour is None:
            return None

        x, y, w, h = cv2.boundingRect(board_contour)
        self.board_region = (x, y, w, h)
        return (x, y, w, h)

    def get_cell_state(self, cell_roi: np.ndarray) -> int:
        """Analyze a cell ROI and return its state: 0=empty, 1=X, 2=O."""
        hsv_cell = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)

        x_mask = cv2.inRange(hsv_cell, X_LOWER_HSV, X_UPPER_HSV)
        o_mask = cv2.inRange(hsv_cell, O_LOWER_HSV, O_UPPER_HSV)

        x_pixels = cv2.countNonZero(x_mask)
        o_pixels = cv2.countNonZero(o_mask)

        threshold = (cell_roi.shape[0] * cell_roi.shape[1]) * 0.05

        if x_pixels > threshold:
            return 1
        if o_pixels > threshold:
            return 2
        return 0

    def detect_board_state(self, frame: np.ndarray) -> list[int]:
        """Detect the current state of all 9 cells.
        Returns list of 9 ints: 0=empty, 1=X (user), 2=O (robot)."""
        board_region = self.find_board_region(frame)
        if board_region is None:
            return [0] * 9

        x, y, w, h = board_region
        board_roi = frame[y : y + h, x : x + w]

        cell_h = h // GRID_DIVISIONS
        cell_w = w // GRID_DIVISIONS

        board_state = []
        for row in range(GRID_DIVISIONS):
            for col in range(GRID_DIVISIONS):
                cell_roi = board_roi[
                    row * cell_h : (row + 1) * cell_h,
                    col * cell_w : (col + 1) * cell_w,
                ]
                state = self.get_cell_state(cell_roi)
                board_state.append(state)

        return board_state

    def detect_new_move(self, frame: np.ndarray) -> Optional[int]:
        """Detect if user made a new move by comparing board states.
        Returns cell position (1-9) or None if no new move detected."""
        current_state = self.detect_board_state(frame)

        for i in range(9):
            if self.previous_board_state[i] == 0 and current_state[i] == 1:
                self.previous_board_state = current_state
                return i + 1

        self.previous_board_state = current_state
        return None

    def draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw the detected board grid on the frame for visualization."""
        if self.board_region is None:
            return frame

        frame_copy = frame.copy()
        x, y, w, h = self.board_region

        for i in range(1, GRID_DIVISIONS):
            cv2.line(
                frame_copy,
                (x + i * w // GRID_DIVISIONS, y),
                (x + i * w // GRID_DIVISIONS, y + h),
                (0, 255, 0),
                2,
            )
            cv2.line(
                frame_copy,
                (x, y + i * h // GRID_DIVISIONS),
                (x + w, y + i * h // GRID_DIVISIONS),
                (0, 255, 0),
                2,
            )

        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame_copy


def communicate_with_logic(move: int) -> Optional[str]:
    """Send move to tic_tac_logic and get response."""
    try:
        response = subprocess.check_output(
            [sys.executable, "tic_tac_logic.py"],
            input=f"{move}\n".encode(),
            timeout=5,
        )
        return response.decode().strip()
    except subprocess.TimeoutExpired:
        print("Timeout communicating with tic_tac_logic")
        return None
    except Exception as e:
        print(f"Error communicating with tic_tac_logic: {e}")
        return None


def run_vision_interface(display: bool = True):
    """Main loop for vision-based game interface.

    Args:
        display: If True, show live camera feed with grid overlay.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    detector = BoardDetector()
    print("Vision interface started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        new_move = detector.detect_new_move(frame)
        if new_move is not None:
            print(f"Detected user move at position {new_move}")
            response = communicate_with_logic(new_move)
            if response:
                print(f"Robot response: {response}")

        if display:
            display_frame = detector.draw_grid(frame)
            cv2.imshow("Tic-Tac-Toe Vision Interface", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_vision_interface(display=True)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
