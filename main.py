import time
import cv2

from vision_interface import BoardDetector
from tic_tac_logic import (
    process_user_move,
    game_status,
    reset_board,
)
from robodk_controller import(move_home)

from robodk_controller import draw_o


ROBOT_BUSY_DELAY = 2.0

# False = camera mode
# True = keyboard mode
USE_KEYBOARD_INPUT = False


# ==========================================
# MANUAL KEYBOARD INPUT
# ==========================================
def get_keyboard_move(detector):

    while True:

        try:
            move = int(input("Enter human move (1-9): "))

            if move < 1 or move > 9:
                print("Move must be between 1 and 9")
                continue

            if move in detector.used_cells:
                print("Cell already used")
                continue

            detector.used_cells.add(move)

            return move

        except ValueError:
            print("Please enter a valid number")


def main():

    global USE_KEYBOARD_INPUT

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = BoardDetector()
    
    move_home()

    print("=== Tic Tac Toe Robot Started ===")
    print("SPACE -> capture move")
    print("K -> keyboard mode")
    print("C -> camera mode")
    print("R -> reset")
    print("Q -> quit")

    robot_busy = False
    last_robot_time = 0
    

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time()

        # Wait while robot moves
        if robot_busy:

            if current_time - last_robot_time > ROBOT_BUSY_DELAY:
                robot_busy = False

        # Draw grid
        display = detector.draw_grid(frame)

        cv2.imshow("Game", display)

        key = cv2.waitKey(1)

        # Quit
        if key & 0xFF == ord('q'):
            break

        # Reset
        if key & 0xFF == ord('r'):

            reset_board()

            detector.reset()

            print("Manual reset")

        # Keyboard mode
        if key & 0xFF == ord('k'):

            USE_KEYBOARD_INPUT = True

            print("Keyboard input mode enabled")

        # Camera mode
        if key & 0xFF == ord('c'):

            USE_KEYBOARD_INPUT = False

            print("Camera detection mode enabled")

        # Capture move
        if key & 0xFF == ord(' '):

            print("Capturing board...")

            time.sleep(0.5)

            snapshot = frame.copy()

            try:

                # ==========================================
                # INPUT SOURCE
                # ==========================================
                if USE_KEYBOARD_INPUT:

                    new_move = get_keyboard_move(detector)

                else:

                    new_move = detector.detect_new_move(snapshot)

                # ==========================================
                # PROCESS MOVE
                # ==========================================
                if new_move is not None:

                    print(f"Human move: {new_move}")

                    robot_move = process_user_move(new_move)

                    status = game_status()

                    print(f"Game status: {status}")

                    if robot_move is not None:

                        robot_busy = True
                        last_robot_time = time.time()

                        draw_o(robot_move)

                        detector.register_robot_move(robot_move)

                        print(f"Robot played: {robot_move}")

                    if status != "in_progress":

                        print("Game finished")

                        time.sleep(3)

                        reset_board()

                        detector.reset()

                        print("New game started")

                else:

                    print("No new move detected")

            except Exception as e:

                print(f"Error: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()