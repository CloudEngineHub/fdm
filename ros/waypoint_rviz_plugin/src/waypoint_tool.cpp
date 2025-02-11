
// CODE MODIFIED FROM: https://github.com/HongbiaoZ/autonomous_exploration_development_environment

#include "waypoint_tool.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace rviz
{
WaypointTool::WaypointTool()
{
  shortcut_key_ = 'w';

  topic_property_ = new StringProperty("Topic", "waypoint", "The topic on which to publish navigation waypionts.",
                                       getPropertyContainer(), SLOT(updateTopic()), this);
}

void WaypointTool::onInitialize()
{
  PoseTool::onInitialize();
  setName("Waypoint");
  updateTopic();
  vehicle_z = 0;
}

void WaypointTool::updateTopic()
{
  const std::string odom_topic = "/state_estimator/pose_in_odom";
  sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped> (odom_topic, 5, &WaypointTool::odomHandler, this);
  pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/mp_waypoint", 5);
  pub_joy_ = nh_.advertise<sensor_msgs::Joy>("/joy", 5);
}

void WaypointTool::odomHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr& odom)
{
  vehicle_z = odom->pose.pose.position.z;
}

void WaypointTool::onPoseSet(double x, double y, double theta)
{
  sensor_msgs::Joy joy;

  joy.axes.push_back(0);
  joy.axes.push_back(0);
  joy.axes.push_back(-1.0);
  joy.axes.push_back(0);
  joy.axes.push_back(1.0);
  joy.axes.push_back(1.0);
  joy.axes.push_back(0);
  joy.axes.push_back(0);

  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(1);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);

  joy.header.stamp = ros::Time::now();
  joy.header.frame_id = "waypoint_tool";
  pub_joy_.publish(joy);
  geometry_msgs::PoseStamped waypoint_odom;
  waypoint_odom.header.frame_id = "odom";
  waypoint_odom.header.stamp = joy.header.stamp;
  waypoint_odom.pose.position.x = x;
  waypoint_odom.pose.position.y = y;
  waypoint_odom.pose.position.z = vehicle_z;
  tf2::Quaternion quat;
  quat.setRPY(0, 0, theta);
  waypoint_odom.pose.orientation = tf2::toMsg(quat);

  // Create a TransformListener object to receive transforms
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  // Wait for the transform to become available
  try {
      // Allow some time for the listener to get the first transform
      ros::Duration(1.0).sleep();

      geometry_msgs::TransformStamped transform = tf_buffer.lookupTransform("map", "odom", ros::Time(0), ros::Duration(1.0));
      geometry_msgs::PoseStamped waypoint_map;
      tf2::doTransform(waypoint_odom, waypoint_map, transform);
      waypoint_map.header.frame_id = "map";
      waypoint_map.header.stamp = ros::Time::now();

      // Print out the transformed pose coordinates
      ROS_INFO("Pose in map frame: (%.2f, %.2f, %.2f, %.2f)",
                waypoint_map.pose.position.x, waypoint_map.pose.position.y, waypoint_map.pose.position.z, theta);

      pub_.publish(waypoint_map);
  } catch (tf2::TransformException &ex) {
      ROS_WARN("%s", ex.what());
      pub_.publish(waypoint_odom);
  }
}
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz::WaypointTool, rviz::Tool)
