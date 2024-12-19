# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 导入同步冒烟测试相关模块
from . import SyncSmokeTest
# 导入冒烟测试相关模块
from . import SmokeTest
# 导入CARLA库，用于与CARLA模拟器交互
import carla
# 导入时间模块，可用于实现例如等待、计时等功能
import time
# 导入数学模块，用于进行数学运算
import math
# 导入numpy库，用于处理数值计算，例如数组操作等
import numpy as np
#导入枚举类相关模块，用于定义枚举类型
from enum import Enum
# 导入队列模块，用于创建队列来存储数据等操作
from queue import Queue
# 导入Empty异常类，用于处理从队列取数据时队列为空的情况
from queue import Empty
# 定义一个枚举类型SensorType，表示传感器类型，这里有两种类型：LIDAR（激光雷达）和SEMLIDAR（语义激光雷达）
class SensorType(Enum):
    LIDAR = 1
    SEMLIDAR = 2
# 定义Sensor类，用于表示一个传感器
class Sensor():
    def __init__(self, test, sensor_type, attributes, sensor_name = None, sensor_queue = None):
        # 保存传入的测试对象，可能用于后续调用测试相关的功能或者获取相关上下文信息
        self.test = test
        # 获取测试对象关联的世界对象（在CARLA中代表整个模拟世界）
        self.world = test.world
        # 记录传感器的类型，通过传入的SensorType枚举值来指定
        self.sensor_type = sensor_type
        # 用于记录传感器出现的错误信息，如果有错误会在这里保存相关的错误描述
        self.error = None
        # 传感器的名称，如果在创建时传入了自定义名称则使用传入的，否则为None
        self.name = sensor_name
        # 用于存储传感器数据的队列，如果创建时传入了相应队列则使用传入的，否则为None
        self.queue = sensor_queue
        # 当前检测到的点数，初始化为0，可能用于后续统计传感器检测到的数据点数等情况
        self.curr_det_pts = 0
        # 根据传入的传感器类型来获取对应的传感器蓝图（blueprint）
        if self.sensor_type == SensorType.LIDAR:
            # 从世界对象的蓝图库中筛选出激光雷达（ray_cast类型）的蓝图，并取第一个（通常只有一个符合条件的）
            self.bp_sensor = self.world.get_blueprint_library().filter("sensor.lidar.ray_cast")[0]
        elif self.sensor_type == SensorType.SEMLIDAR:
            # 同理，获取语义激光雷达（ray_cast_semantic类型）的蓝图
            self.bp_sensor = self.world.get_blueprint_library().filter("sensor.lidar.ray_cast_semantic")[0]
        else:
            # 如果传入的传感器类型不在定义的枚举类型中，设置错误信息
            self.error = "Unknown type of sensor"
        # 遍历传入的传感器属性字典，为传感器蓝图设置相应的属性
        for key in attributes:
            self.bp_sensor.set_attribute(key, attributes[key])
        # 获取地图中的出生点列表，并取第一个出生点作为传感器的初始位置变换信息
        tranf = self.world.get_map().get_spawn_points()[0]
        # 将传感器的初始位置在Z轴方向上抬高3个单位（可能是为了让传感器处于合适的高度位置来检测）
        tranf.location.z += 3
        # 在世界中根据传感器蓝图和设置好的位置变换信息生成具体的传感器实例（actor）
        self.sensor = self.world.spawn_actor(self.bp_sensor, tranf)
        # 让传感器开始监听数据，当有数据到来时，会调用callback函数进行处理，
        # 并传入传感器数据、传感器名称以及存储数据的队列作为参数
        self.sensor.listen(lambda sensor_data: self.callback(sensor_data, self.name, self.queue))

    def destroy(self): # 这里假设存在一个包含这些方法的类，原代码中未明确给出类名，需根据实际情况调整
        self.sensor.destroy()
        """
                用于销毁传感器相关资源的方法，调用传感器对象自身的destroy方法来执行具体的销毁操作。
                """

    def callback(self, sensor_data, sensor_name=None, queue=None):
        # Compute the total sum of points adding all channels
        total_channel_points = 0
        for i in range(0, sensor_data.channels):
            total_channel_points += sensor_data.get_point_count(i)
        """
               回调函数，用于处理传感器数据并进行一些相关的计算和验证操作。

               参数:
               - sensor_data: 传感器采集的数据对象，包含了如通道数、原始数据等信息。
               - sensor_name: 传感器的名称（可选参数，可为None）。
               - queue: 用于同步数据的队列（可选参数，可为None）。
               """
        total_detect_points = 0
        for _detection in sensor_data:
            total_detect_points += 1

        # Point cloud used with numpy from the raw data
        if self.sensor_type == SensorType.LIDAR:
            """
            如果传感器类型是普通LiDAR，从传感器数据的原始数据缓冲区创建numpy数组，
            并进行形状重塑等操作，同时记录当前处理后的点数量。
            """
            # Compute the total sum of points adding all channels
            points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            total_np_points = points.shape[0]
            self.curr_det_pts = total_np_points
        elif self.sensor_type == SensorType.SEMLIDAR:
            """
                       如果传感器类型是语义LiDAR（SEMLIDAR），从传感器数据的原始数据缓冲区按照特定的数据类型结构创建numpy数组，
                       提取相关坐标信息组成点云数据，记录点数量并赋值给当前检测点数量属性。
                       """
            data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
            points = np.array([data['x'], data['y'], data['z']]).T
            total_np_points = points.shape[0]
            self.curr_det_pts = total_np_points
        else:
            self.error = "It should never reach this point"
            return

        if total_np_points != total_detect_points:
            self.error = "The number of points of the raw data does not match with the LidarMeasurament array"

        if total_channel_points != total_detect_points:
            self.error = "The sum of the points of all channels does not match with the LidarMeasurament array"

        # Add option to synchronization queue
        if queue is not None:
            queue.put((sensor_data.frame, sensor_name, self.curr_det_pts))

    def is_correct(self):
        """
              判断传感器数据处理过程是否正确，通过检查是否有错误信息来确定，若error属性为None则表示正确。
              """
        return self.error is None

    def get_current_detection_points():
        """
               获取当前检测到的点的数量，返回self.curr_det_pts的值（此处代码可能有问题，应该添加self参数，如def get_current_detection_points(self)，按实际需求调整）。
               """
        return self.curr_det_pts
# 以下是不同类型的测试类，用于针对不同情况对传感器相关功能进行测试
# 测试同步LiDAR传感器点数量的类，继承自SyncSmokeTest，包含了具体的测试方法
class TestSyncLidar(SyncSmokeTest):
    def test_lidar_point_count(self):
        """
               测试同步LiDAR传感器点数量的方法，主要步骤包括创建多个LiDAR传感器对象、模拟世界时间推进、销毁传感器以及检查传感器处理过程是否正确。
               """
        print("TestSyncLidar.test_lidar_point_count")
        sensors = []
# 定义第一个激光雷达传感器的属性字典，包含通道数、强度衰减限制、一般衰减率、检测范围、每秒点数、旋转频率等属性
        att_l00={'channels' : '64', 'dropoff_intensity_limit': '0.0', 'dropoff_general_rate': '0.0',
          'range' : '50', 'points_per_second': '100000', 'rotation_frequency': '20'}
        # 定义第二个激光雷达传感器的属性字典
        att_l01={'channels' : '64', 'range' : '200', 'points_per_second': '500000',
          'rotation_frequency': '5'}
        # 定义第三个激光雷达传感器的属性字典
        att_l02={'channels' : '64', 'dropoff_intensity_limit': '1.0', 'dropoff_general_rate': '0.0',
          'range' : '50', 'points_per_second': '100000', 'rotation_frequency': '50'}
         # 创建三个不同属性配置的激光雷达传感器对象，并添加到传感器列表中
        sensors.append(Sensor(self, SensorType.LIDAR, att_l00))
        sensors.append(Sensor(self, SensorType.LIDAR, att_l01))
        sensors.append(Sensor(self, SensorType.LIDAR, att_l02))
        # 模拟世界时间推进10次，让传感器有足够时间采集数据
        for _ in range(0, 10):
            self.world.tick()
            # 暂停程序0.5秒，等待一些可能的后台处理完成，确保数据稳定
        time.sleep(0.5)
        # 销毁每个传感器对象，释放相关资源
        for sensor in sensors:
            sensor.destroy()
        # 检查每个传感器的数据处理过程是否正确，如果有错误则使测试失败并输出错误信息
        for sensor in sensors:
            if not sensor.is_correct():
                self.fail(sensor.error)


    def test_semlidar_point_count(self):
        """
               测试同步语义LiDAR（SEMLIDAR）传感器点数量的方法，流程与测试同步LiDAR类似，包括创建传感器、模拟世界时间推进、销毁传感器以及检查正确性。
               """
        print("TestSyncLidar.test_semlidar_point_count")
        sensors = []
         # 定义第一个语义激光雷达传感器的属性字典，包含通道数、检测范围、每秒点数、旋转频率等属性
        att_s00 = {'channels' : '64', 'range' : '100', 'points_per_second': '100000',
          'rotation_frequency': '20'}
        # 定义第二个语义激光雷达传感器的属性字典
        att_s01 = {'channels' : '32', 'range' : '200', 'points_per_second': '500000',
          'rotation_frequency': '50'}
        # 创建两个不同属性配置的语义激光雷达传感器对象，并添加到传感器列表中
        sensors.append(Sensor(self, SensorType.SEMLIDAR, att_s00))
        sensors.append(Sensor(self, SensorType.SEMLIDAR, att_s01))
        # 模拟世界时间推进10次，让传感器有足够时间采集数据
        for _ in range(0, 10):
            self.world.tick()
            # 暂停程序0.5秒，等待一些可能的后台处理完成，确保数据稳定
        time.sleep(0.5)
        # 销毁每个传感器对象，释放相关资源
        for sensor in sensors:
            sensor.destroy()
        # 检查每个传感器的数据处理过程是否正确，如果有错误则使测试失败并输出错误信息
        for sensor in sensors:
            if not sensor.is_correct():
                self.fail(sensor.error)


class TestASyncLidar(SmokeTest):
    """
            测试异步LiDAR传感器点数量的方法，创建多个LiDAR传感器对象后等待一段时间，然后销毁传感器并检查处理过程是否正确。
            """
    def test_lidar_point_count(self):
        print("TestASyncLidar.test_lidar_point_count")
        sensors = []
        # 定义第一个激光雷达传感器的属性字典，包含通道数、强度衰减限制、一般衰减率、检测范围、每秒点数、旋转频率等属性
        att_l00={'channels' : '64', 'dropoff_intensity_limit': '0.0', 'dropoff_general_rate': '0.0',
          'range' : '50', 'points_per_second': '100000', 'rotation_frequency': '20'}
        att_l01={'channels' : '64', 'range' : '200', 'points_per_second': '500000',
          'rotation_frequency': '5'}
        att_l02={'channels' : '64', 'dropoff_intensity_limit': '1.0', 'dropoff_general_rate': '0.0',
          'range' : '50', 'points_per_second': '100000', 'rotation_frequency': '50'}
        #将三个激光雷达传感器添加到传感器列表中
        sensors.append(Sensor(self, SensorType.LIDAR, att_l00))
        sensors.append(Sensor(self, SensorType.LIDAR, att_l01))
        sensors.append(Sensor(self, SensorType.LIDAR, att_l02))

        time.sleep(3.0)
        #销毁所有传感器
        for sensor in sensors:
            sensor.destroy()
        #检查每个传感器是否正确，如果不正确则标记测试失败
        for sensor in sensors:
            if not sensor.is_correct():
                self.fail(sensor.error)


    def test_semlidar_point_count(self):
        """
               比较不同类型LiDAR（语义LiDAR、普通LiDAR等）的方法，主要操作包括创建不同类型的LiDAR传感器对象并放入队列、模拟世界时间推进、从队列获取数据进行比较验证等。
               """
        print("TestASyncLidar.test_semlidar_point_count")
        sensors = []
        # 定义第一个激光雷达传感器的属性字典，包含通道数、强度衰减限制、一般衰减率、检测范围、每秒点数、旋转频率等属性
        att_s00 = {'channels' : '64', 'range' : '100', 'points_per_second': '100000',
          'rotation_frequency': '20'}
        # 定义第二个语义激光雷达传感器的属性字典
        att_s01 = {'channels' : '32', 'range' : '200', 'points_per_second': '500000',
          'rotation_frequency': '50'}
        # 创建两个不同属性配置的语义激光雷达传感器对象，并添加到传感器列表中
        sensors.append(Sensor(self, SensorType.SEMLIDAR, att_s00))
        sensors.append(Sensor(self, SensorType.SEMLIDAR, att_s01))

        time.sleep(3.0)
        #销毁所有传感器
        for sensor in sensors:
            sensor.destroy()
        
        for sensor in sensors:
            if not sensor.is_correct():
                self.fail(sensor.error)

class TestCompareLidars(SyncSmokeTest):
    def test_lidar_comparison(self):
        print("TestCompareLidars.test_lidar_comparison")
        sensors = []

        att_sem_lidar={'channels' : '64', 'range' : '200', 'points_per_second': '500000'}
        att_lidar_nod={'channels' : '64', 'dropoff_intensity_limit': '0.0', 'dropoff_general_rate': '0.0',
          'range' : '200', 'points_per_second': '500000'}
        att_lidar_def={'channels' : '64', 'range' : '200', 'points_per_second': '500000'}

        sensor_queue = Queue()
        sensors.append(Sensor(self, SensorType.SEMLIDAR, att_sem_lidar, "SemLidar", sensor_queue))
        sensors.append(Sensor(self, SensorType.LIDAR, att_lidar_nod, "LidarNoD", sensor_queue))
        sensors.append(Sensor(self, SensorType.LIDAR, att_lidar_def, "LidarDef", sensor_queue))

        for _ in range(0, 15):
            self.world.tick()

            data_sem_lidar = None
            data_lidar_nod = None
            data_lidar_def = None
            for _ in range(len(sensors)):
                data = sensor_queue.get(True, 10.0)
                if data[1] == "SemLidar":
                    data_sem_lidar = data
                elif data[1] == "LidarNoD":
                    data_lidar_nod = data
                elif data[1] == "LidarDef":
                    data_lidar_def = data
                else:
                    self.fail("It should never reach this point")

            # 检查帧序号是否正确
            self.assertEqual(data_sem_lidar[0], data_lidar_nod[0], "The frame numbers of LiDAR and SemLiDAR do not match.")
            self.assertEqual(data_sem_lidar[0], data_lidar_def[0], "The frame numbers of LiDAR and SemLiDAR do not match.")

            # The detections of the semantic lidar and the Lidar with no dropoff should have the same point count always
            self.assertEqual(data_sem_lidar[2], data_lidar_nod[2], "The point count of the detections of this frame of LiDAR(No dropoff) and SemLiDAR do not match.")

            # Default lidar should drop a minimum of 45% of the points so we check that but with a high tolerance to account for 'rare' cases
            if data_lidar_def[2] > 0.75 * data_sem_lidar[2]:
                self.fail("The point count of the default lidar should be much less than the Semantic Lidar point count.")

        time.sleep(1)
        for sensor in sensors:
            sensor.destroy()
