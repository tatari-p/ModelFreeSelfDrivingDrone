from AirSimClient import *


def amp(vec):
    return np.sqrt(np.sum(vec**2))


def unit(vec):
    return vec/amp(vec)


class ModelFreeAirSimEnv:
    _target = None
    _init = np.array([0, 0, 0], dtype="float")

    def __init__(self, client, scale=1., goal_range=10.):
        self.client = client
        self.scale = scale
        self.goal_range = goal_range

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = np.array(value, dtype="float")

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, value):
        self._init = np.array(value, dtype="float")

    def set_to_drive(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.takeoff()
        self.client.moveToZ(-10., 5)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.init = [0, 0, 0]
        self.takeoff()
        self.client.moveToZ(-10., 5)
        time.sleep(0.5)

    def takeoff(self):
        self.client.takeoff()
        time.sleep(0.5)

    def hover(self):
        self.client.hover()
        time.sleep(3)

    def get_position(self):
        p = self.client.getPosition()

        return np.array([p.x_val, p.y_val, p.z_val], dtype="float")

    def get_velocity(self):
        v = self.client.getVelocity()

        return np.array([v.x_val, v.y_val, v.z_val], dtype="float")

    def get_orient(self):
        orient = self.client.getOrientation()

        return np.array([orient.w_val, orient.x_val, orient.y_val, orient.z_val], dtype="float")

    def head_to_target(self):
        z = self.get_position()[2]
        head = self.target-self.init
        angle = np.arctan2(head[1], head[0])

        self.client.moveByAngle(0., 0., z, angle, 10)
        time.sleep(3)

    def drive(self, action):
        v_before = self.get_velocity()
        head = unit(self.target-self.init)
        v_drive = unit(action)
        v_drive = v_drive[2]*np.array([0, 0, 1], dtype="float")+v_drive[1]*head+v_drive[0]*np.array([-head[1], head[0], 0], dtype="float")

        forward = self.target-self.get_position()
        v_offset = unit(unit(forward)-unit(v_before))
        v_after = self.scale*(v_offset+v_drive)+v_before

        self.client.moveByVelocity(*v_after, 5)
        time.sleep(0.5)

    def get_state(self):
        view = self.client.simGetImages([ImageRequest(0, AirSimImageType.DepthPlanner, True, False)])[0]
        view_height = view.height
        view_width = view.width
        view = np.reshape(np.clip(view.image_data_float, 0, 100), [view_height, view_width, 1])

        v = self.get_velocity()
        head = self.target-self.init
        v_z = v[2]
        angle = np.arctan2(head[1], head[0])
        v_x = np.cos(angle)*v[0]+np.sin(angle)*v[1]
        v_y = -np.sin(angle)*v[0]+np.cos(angle)*v[1]
        v_head = np.array([v_x, v_y, v_z], dtype="float32")
        p = self.get_position()
        d = self.target-p
        d_x = np.cos(angle)*d[0]+np.sin(angle)*d[1]
        d_y = -np.sin(angle)*d[0]+np.cos(angle)*d[1]
        d_head = np.array([d_x, d_y, p[2]], dtype="float32")

        state = (view, v_head, d_head)

        collision_info = self.client.getCollisionInfo()
        done = False

        s = amp(d[:2])

        if collision_info.has_collided:
            reward = -1.
            done = True
            print("*** Drone was collided with", collision_info.object_name, "at %.1f from goal" % s)
            self.reset()

        elif s < self.goal_range:
            reward = 1.
            done = True
            print("*** Goal in at (%.1f, %.1f, %.1f)" % (p[0], p[1], p[2]))
            self.hover()
            self.init = p
            time.sleep(0.5)

        else:
            reward = -0.005

        return state, reward, done