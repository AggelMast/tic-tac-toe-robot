import time
import cv2

from vision_interface import BoardDetector
from tic_tac_logic import (
    process_user_move,
    game_status,
    reset_board,
)

from robodk_controller import draw_o

ROBOT_BUSY_DELAY = 2.0


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = BoardDetector()

    print("=== Tic Tac Toe Robot Started ===")
    print("SPACE -> capture move")
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
            detector.previous_board_state = [0] * 9

            print("Manual reset")

        # Capture screenshot for move detection
        if key & 0xFF == ord(' '):

            print("Capturing board...")
            
            time.sleep(0.5)

            snapshot = frame.copy()

            try:

                new_move = detector.detect_new_move(snapshot)

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

                        #detector.previous_board_state = [0] * 9
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