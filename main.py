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

            cv2.imshow("Game", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # Detect human move
        new_move = detector.detect_new_move(frame)

        if new_move is not None:

            print(f"Human move: {new_move}")

            try:

                robot_move = process_user_move(new_move)

                status = game_status()

                print(f"Game status: {status}")

                if robot_move is not None:

                    robot_busy = True
                    last_robot_time = time.time()

                    draw_o(robot_move)

                    print(f"Robot played: {robot_move}")

                if status != "in_progress":

                    print("Game finished")

                    time.sleep(3)

                    reset_board()

                    detector.previous_board_state = [0] * 9

                    print("New game started")

            except Exception as e:
                print(f"Error: {e}")

        # Visualization
        display = detector.draw_grid(frame)

        cv2.imshow("Game", display)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('r'):

            reset_board()
            detector.previous_board_state = [0] * 9

            print("Manual reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()