from miniarm.controller import MiniArmVelocityController
from miniarm.interactive import run_interactive

def main():
    arm = MiniArmVelocityController()
    try:
        run_interactive(arm)
    except KeyboardInterrupt:
        print("\nInterrupted")
    print("Finished")


if __name__ == "__main__":
    controller = MiniArmVelocityController()
    run_interactive(controller)