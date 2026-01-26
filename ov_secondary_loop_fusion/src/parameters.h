/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>

extern camodocal::CameraPtr m_camera;
extern double max_focallength;
extern double MIN_SCORE;
extern double PNP_INFLATION;
extern int RECALL_IGNORE_RECENT_COUNT;
extern double MAX_THETA_DIFF;
extern double MAX_POS_DIFF;
extern int MIN_LOOP_NUM;
extern double MIN_OPTIMIZATION_TIME_DIFF;
extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;
extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern std::string POSE_GRAPH_LOAD_PATH;
extern std::string LOOP_RESULT_FOLDER;
extern int ROW;
extern int COL;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_RESULT_FOLDER;
extern std::string VINS_ALL_TRAJ_FOLDER;
extern int DEBUG_IMAGE;
extern bool SAVE_CAM_POSES;
