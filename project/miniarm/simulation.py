import pybullet as p
import pybullet_data

def init_simulation(gui=True):
    physicsClient = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 讓你原本的本地 URDF 路徑也被搜尋到
    p.setAdditionalSearchPath("./mini_arm_URDF_V14")
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    try:
        planeId = p.loadURDF("plane.urdf")
    except:
        print("No ground plane loaded")
        planeId = None
    return physicsClient, planeId

def load_robot():
    urdf_paths = [
        "urdf/mini_arm_URDF_V14.urdf",
        "mini_arm_URDF_V14.urdf",
        "mini_arm.urdf"
    ]
    for urdf in urdf_paths:
        try:
            robotId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
            print(f"Loaded Mini Arm from {urdf}")
            return robotId
        except:
            continue
    print("Using KUKA as fallback")
    return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
