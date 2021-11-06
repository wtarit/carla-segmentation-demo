# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
import carla

import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
from itertools import groupby


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


best_model = torch.load("best_model.pth")
ENCODER = 'timm-mobilenetv3_large_075'
ENCODER_WEIGHTS = 'imagenet'


preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


IM_WIDTH = 640
IM_HEIGHT = 480


class Car:
    def __init__(self) -> None:
        self.steer = 0
        self.actor_list = []

        client = carla.Client('192.168.1.109', 2000)
        client.set_timeout(5.0)

        world = client.load_world('Town07')

        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter('model3')[0]
        spawn_point = carla.Transform(carla.Location(
            x=-9.188781, y=-239.017456, z=0.3), carla.Rotation(pitch=0.231680, yaw=-166.279816, roll=0.967588))

        self.vehicle = world.spawn_actor(bp, spawn_point)

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, steer=0.0))

        self.actor_list.append(self.vehicle)

        rgb_cam = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        rgb_cam.set_attribute('image_size_x', f'{IM_WIDTH}')
        rgb_cam.set_attribute('image_size_y', f'{IM_HEIGHT}')
        rgb_cam.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2, z=1.5))

        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(
            rgb_cam, spawn_point, attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(sensor)

        # Process Sensor Data
        sensor.listen(lambda data: self.process_img(data))

        third_person_view = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        third_person_view.set_attribute('image_size_x', f'{1280}')
        third_person_view.set_attribute('image_size_y', f'{720}')

        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=-7.5, z=2.5),
                                      carla.Rotation(pitch=8.0))

        # spawn the sensor and attach to vehicle.
        tpv_cam = world.spawn_actor(
            third_person_view, spawn_point, attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(tpv_cam)
        tpv_cam.listen(lambda data: self.show_tpv(data))

        input()

    def show_tpv(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((720, 1280, 4))
        image = i[:, :, :3]
        cv2.imshow("car", image)
        cv2.waitKey(1)

    def process_img(self, image):
        image = np.array(image.raw_data)
        image = image.reshape((IM_HEIGHT, IM_WIDTH, 4))
        image = image[:, :, :3]
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)
        image = to_tensor(image)
        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        t_range = []

        i = 0
        for k, g in groupby(pr_mask[300]):
            l = len(list(g))
            if k:
                t_range.append((i, i+l))
            i += l
        # print(t_range)
        if t_range is not None:
            t_range = max(t_range, key=lambda l: l[1] - l[0])
            mid = (t_range[0] + t_range[1]) / 2
            self.steer = (mid / 320) - 1
            self.steer = self.steer - 0.2
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.3, steer=self.steer))

        orig_image = cv2.circle(orig_image, (int(mid), 300),
                                5, (0, 0, 255), cv2.FILLED)

        cv2.imshow("mask", pr_mask)
        cv2.imshow("img", orig_image)
        cv2.waitKey(1)

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')


try:
    car = Car()

finally:
    car.destroy()
