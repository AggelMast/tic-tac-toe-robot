from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# ==========================================
# BOARD POSITION
# ==========================================

BOARD_X = 180
BOARD_Y = 100

BOARD_W = 240
BOARD_H = 240

GRID_SIZE = 3


class BoardDetector:

    def __init__(self):

        # current detected board
        self.previous_board_state = [0] * 9

        # robot played cells
        self.robot_cells = set()

        # all occupied cells
        self.used_cells = set()

    # ==========================================
    # ADD ROBOT MOVE
    # ==========================================
    def register_robot_move(self, cell):

        self.robot_cells.add(cell)

        self.used_cells.add(cell)

    # ==========================================
    # RESET
    # ==========================================
    def reset(self):

        self.previous_board_state = [0] * 9

        self.robot_cells.clear()

        self.used_cells.clear()

    # ==========================================
    # EXTRACT BOARD
    # ==========================================
    def extract_board(self, frame):

        return frame[
            BOARD_Y:BOARD_Y + BOARD_H,
            BOARD_X:BOARD_X + BOARD_W
        ]

    # ==========================================
    # PREPROCESS
    # ==========================================
    def preprocess(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, binary = cv2.threshold(
            blur,
            120,
            255,
            cv2.THRESH_BINARY_INV
        )

        return binary

    # ==========================================
    # DETECT CELL
    # ==========================================
    def detect_cell(self, cell_binary):

        black_pixels = cv2.countNonZero(cell_binary)

        area = (
            cell_binary.shape[0]
            * cell_binary.shape[1]
        )

        percentage = black_pixels / area

        print("Percentage:", percentage)

        if percentage > 0.08:
            return 1

        return 0

    # ==========================================
    # DETECT BOARD
    # ==========================================
    def detect_board_state(self, frame):

        board = self.extract_board(frame)

        binary = self.preprocess(board)

        h, w = binary.shape

        cell_h = h // GRID_SIZE
        cell_w = w // GRID_SIZE

        board_state = []

        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                x1 = col * cell_w
                y1 = row * cell_h

                x2 = x1 + cell_w
                y2 = y1 + cell_h

                cell = binary[y1:y2, x1:x2]

                state = self.detect_cell(cell)

                board_state.append(state)

        print("Board State:", board_state)

        return board_state

    # ==========================================
    # DETECT NEW HUMAN MOVE
    # ==========================================
    def detect_new_move(self, frame) -> Optional[int]:

        current_state = self.detect_board_state(frame)

        for i in range(9):

            cell_number = i + 1

            # already occupied -> ignore
            if cell_number in self.used_cells:
                continue

            # new X detected
            if current_state[i] == 1:

                self.used_cells.add(cell_number)

                self.previous_board_state = current_state

                return cell_number

        self.previous_board_state = current_state

        return None

    # ==========================================
    # DRAW GRID
    # ==========================================
    def draw_grid(self, frame):

        frame = frame.copy()

        x = BOARD_X
        y = BOARD_Y
        w = BOARD_W
        h = BOARD_H

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        for i in range(1, GRID_SIZE):

            # vertical
            cv2.line(
                frame,
                (x + i * w // 3, y),
                (x + i * w // 3, y + h),
                (0, 255, 0),
                2
            )

            # horizontal
            cv2.line(
                frame,
                (x, y + i * h // 3),
                (x + w, y + i * h // 3),
                (0, 255, 0),
                2
            )

        return frame