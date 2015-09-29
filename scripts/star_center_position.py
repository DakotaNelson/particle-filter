#!/usr/bin/env python

""" This code implements a ceiling-marker based localization system.
    The core of the code is filling out the marker_locators
    which allow for the specification of the position and orientation
    of the markers on the ceiling of the room """

import rospy
from ar_pose.msg import ARMarkers
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix, quaternion_from_euler
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from tf import TransformListener, TransformBroadcaster
from copy import deepcopy
from math import sin, cos, pi, atan2, fabs

class TransformHelpers:
    """ Some convenience functions for translating between various representions of a robot pose.
        TODO: nothing... you should not have to modify these """

    @staticmethod
    def convert_translation_rotation_to_pose(translation, rotation):
        """ Convert from representation of a pose as translation and rotation (Quaternion) tuples to a geometry_msgs/Pose message """
        return Pose(position=Point(x=translation[0],y=translation[1],z=translation[2]), orientation=Quaternion(x=rotation[0],y=rotation[1],z=rotation[2],w=rotation[3]))

    @staticmethod
    def convert_pose_inverse_transform(pose):
        """ Helper method to invert a transform (this is built into the tf C++ classes, but ommitted from Python) """
        translation = np.zeros((4,1))
        translation[0] = -pose.position.x
        translation[1] = -pose.position.y
        translation[2] = -pose.position.z
        translation[3] = 1.0

        rotation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler_angle = euler_from_quaternion(rotation)
        rotation = np.transpose(rotation_matrix(euler_angle[2], [0,0,1]))       # the angle is a yaw
        transformed_translation = rotation.dot(translation)

        translation = (transformed_translation[0], transformed_translation[1], transformed_translation[2])
        rotation = quaternion_from_matrix(rotation)
        return (translation, rotation)

    @staticmethod
    def angle_normalize(z):
        """ convenience function to map an angle to the range [-pi,pi] """
        return atan2(sin(z), cos(z))

    @staticmethod
    def angle_diff(a, b):
        """ Calculates the difference between angle a and angle b (both should be in radians)
            the difference is always based on the closest rotation from angle a to angle b
            examples:
                angle_diff(.1,.2) -> -.1
                angle_diff(.1, 2*math.pi - .1) -> .2
                angle_diff(.1, .2+2*math.pi) -> -.1
        """
        a = TransformHelpers.angle_normalize(a)
        b = TransformHelpers.angle_normalize(b)
        d1 = a-b
        d2 = 2*pi - fabs(d1)
        if d1 > 0:
            d2 *= -1.0
        if fabs(d1) < fabs(d2):
            return d1
        else:
            return d2

class MarkerLocator(object):
    def __init__(self, id, position, yaw):
        """ Create a MarkerLocator object
            id: the id of the marker (this is an index based on the file
                specified in ar_pose_multi.launch)
            position: this is a tuple of the x,y position of the marker
            yaw: this is the angle about a normal vector pointed towards
                 the STAR center ceiling
        """
        self.id = id
        self.position = position
        self.yaw = yaw

    def get_camera_position(self, marker):
        """ Outputs the position of the camera in the global coordinates """
        euler_angles = euler_from_quaternion((marker.pose.pose.orientation.x,
                                              marker.pose.pose.orientation.y,
                                              marker.pose.pose.orientation.z,
                                              marker.pose.pose.orientation.w))
        translation = np.array([marker.pose.pose.position.y,
                                -marker.pose.pose.position.x,
                                0,
                                1.0])
        translation_rotated = rotation_matrix(self.yaw-euler_angles[2], [0,0,1]).dot(translation)
        xy_yaw = (translation_rotated[0]+self.position[0],translation_rotated[1]+self.position[1],self.yaw-euler_angles[2])
        return xy_yaw

class MarkerProcessor(object):
    def __init__(self, use_dummy_transform=False):
        rospy.init_node('star_center_positioning_node')
        if use_dummy_transform:
            self.odom_frame_name = "odom_dummy"
        else:
            self.odom_frame_name = "odom"

        self.marker_locators = {}
        self.add_marker_locator(MarkerLocator(0,(0.0,0.0),0))
        self.add_marker_locator(MarkerLocator(1,(1.4/1.1,2.0/1.1),0))

        self.marker_sub = rospy.Subscriber("ar_pose_marker",
                                           ARMarkers,
                                           self.process_markers)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.process_odom, queue_size=10)
        self.star_pose_pub = rospy.Publisher("STAR_pose",PoseStamped,queue_size=10)
        self.continuous_pose = rospy.Publisher("STAR_pose_continuous",PoseStamped,queue_size=10)
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

    def add_marker_locator(self, marker_locator):
        self.marker_locators[marker_locator.id] = marker_locator

    def process_odom(self, msg):
        p = PoseStamped(header=Header(stamp=rospy.Time(0), frame_id=self.odom_frame_name),
                        pose=msg.pose.pose)
        try:
            STAR_pose = self.tf_listener.transformPose("STAR", p)
            STAR_pose.header.stamp = msg.header.stamp
            self.continuous_pose.publish(STAR_pose)
        except Exception as inst:
            print "error is", inst

    def process_markers(self, msg):
        for marker in msg.markers:
            # do some filtering basd on prior knowledge
            # we know the approximate z coordinate and that all angles but yaw should be close to zero
            euler_angles = euler_from_quaternion((marker.pose.pose.orientation.x,
                                                  marker.pose.pose.orientation.y,
                                                  marker.pose.pose.orientation.z,
                                                  marker.pose.pose.orientation.w))
            angle_diffs = TransformHelpers.angle_diff(euler_angles[0],pi), TransformHelpers.angle_diff(euler_angles[1],0)
            if (marker.id in self.marker_locators and
                2.4 <= marker.pose.pose.position.z <= 2.6 and
                fabs(angle_diffs[0]) <= .2 and
                fabs(angle_diffs[1]) <= .2):
                locator = self.marker_locators[marker.id]
                xy_yaw = locator.get_camera_position(marker)
                orientation_tuple = quaternion_from_euler(0,0,xy_yaw[2])
                pose = Pose(position=Point(x=xy_yaw[0],y=xy_yaw[1],z=0),
                            orientation=Quaternion(x=orientation_tuple[0], y=orientation_tuple[1], z=orientation_tuple[2], w=orientation_tuple[3]))
                # TODO: use markers timestamp instead of now() (unfortunately, not populated currently by ar_pose)
                pose_stamped = PoseStamped(header=Header(stamp=rospy.Time.now(),frame_id="STAR"),pose=pose)
                try:
                    offset, quaternion = self.tf_listener.lookupTransform("/base_link", "/base_laser_link", rospy.Time(0))
                except Exception as inst:
                    print "Error", inst
                    return
                # TODO: use frame timestamp instead of now()
                pose_stamped_corrected = deepcopy(pose_stamped)
                pose_stamped_corrected.pose.position.x -= offset[0]*cos(xy_yaw[2])
                pose_stamped_corrected.pose.position.y -= offset[0]*sin(xy_yaw[2])
                self.star_pose_pub.publish(pose_stamped_corrected)
                self.fix_STAR_to_odom_transform(pose_stamped_corrected)

    def fix_STAR_to_odom_transform(self, msg):
        """ Super tricky code to properly update map to odom transform... do not modify this... Difficulty level infinity. """
        (translation, rotation) = TransformHelpers.convert_pose_inverse_transform(msg.pose)
        p = PoseStamped(pose=TransformHelpers.convert_translation_rotation_to_pose(translation,rotation),header=Header(stamp=rospy.Time(),frame_id="base_link"))
        try:
            self.tf_listener.waitForTransform("odom","base_link",rospy.Time(),rospy.Duration(1.0))
        except Exception as inst:
            print "whoops", inst
            return
        print "got transform"
        self.odom_to_STAR = self.tf_listener.transformPose("odom", p)
        (self.translation, self.rotation) = TransformHelpers.convert_pose_inverse_transform(self.odom_to_STAR.pose)

    def broadcast_last_transform(self):
        """ Make sure that we are always broadcasting the last map to odom transformation.
            This is necessary so things like move_base can work properly. """
        if not(hasattr(self,'translation') and hasattr(self,'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation, self.rotation, rospy.get_rostime(), self.odom_frame_name, "STAR")

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.broadcast_last_transform()
            r.sleep()

if __name__ == '__main__':
    nh = MarkerProcessor(use_dummy_transform=True)
    nh.run()
