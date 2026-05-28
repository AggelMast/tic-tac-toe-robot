from robodk.robolink import *
from robodk.robomath import *
import time

RDK = Robolink()

robot = RDK.Item('', ITEM_TYPE_ROBOT)
RDK.setRunMode(RUNMODE_RUN_ROBOT)


if not robot.Valid():
    raise Exception("Robot not found in RoboDK")

# Home target
HOME = RDK.Item('Home')

# Mapping cell -> RoboDK target names
CELL_TARGETS = {
    1: ("Cell_1_Approach", "Cell_1_Draw"),
    2: ("Cell_2_Approach", "Cell_2_Draw"),
    3: ("Cell_3_Approach", "Cell_3_Draw"),
    4: ("Cell_4_Approach", "Cell_4_Draw"),
    5: ("Cell_5_Approach", "Cell_5_Draw"),
    6: ("Cell_6_Approach", "Cell_6_Draw"),
    7: ("Cell_7_Approach", "Cell_7_Draw"),
    8: ("Cell_8_Approach", "Cell_8_Draw"),
    9: ("Cell_9_Approach", "Cell_9_Draw"),
}


def move_home():
    robot.setSpeedJoints(10)
    robot.setAccelerationJoints(30)
    robot.setSpeed(30)
    robot.setAcceleration(60)
    robot.MoveL(HOME)


def draw_o(cell: int):

    if cell not in CELL_TARGETS:
        raise ValueError(f"Invalid cell: {cell}")

    approach_name, draw_name = CELL_TARGETS[cell]

    approach = RDK.Item(approach_name)
    draw = RDK.Item(draw_name)

    if not approach.Valid():
        raise Exception(f"Target not found: {approach_name}")

    if not draw.Valid():
        raise Exception(f"Target not found: {draw_name}")

    print(f"Robot drawing O at cell {cell}")

    # Safe approach
    print("approach:", approach)
    robot.MoveL(approach)

    # Move to drawing position
    robot.MoveL(draw)

    # Draw circle
    pose = robot.Pose()

    radius = 10
    points = 20

    for i in range(points + 1):

        angle = (2 * 3.14159 * i) / points

        x = radius * cos(angle)
        y = radius * sin(angle)

        new_pose = transl(x, y, 0) * pose

        robot.MoveL(new_pose)

    # Return safely
    robot.MoveL(approach)

    move_home()