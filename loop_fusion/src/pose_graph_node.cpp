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

#include <vector>
#include <rclcpp/node.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/wait_for_message.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <filesystem>
#include <iostream>
//#include <ros/package.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#include "pose_graph_node.h"


App::App(rclcpp::Node::SharedPtr node, const CommandLineConfig &app_params):
    node_(node), app_params_(app_params) {

    pub_match_img = node_->create_publisher<sensor_msgs::msg::Image>("/match_image", 1000);
    pub_camera_pose_visual = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/camera_pose_visual", 1000);
    pub_point_cloud = node_->create_publisher<sensor_msgs::msg::PointCloud>("/point_cloud_loop_rect", 1000);
    pub_margin_cloud = node_->create_publisher<sensor_msgs::msg::PointCloud>("/margin_cloud_loop_rect", 1000);
    pub_odometry_rect = node_->create_publisher<nav_msgs::msg::Odometry>("/odometry_rect", 1000);
    pub_pose_rect = node_->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_rect", 1000);
}


#define SKIP_FIRST_CNT 2
using namespace std;

queue<sensor_msgs::msg::Image::ConstSharedPtr> image_buf;
std::queue<sensor_msgs::msg::PointCloud::ConstSharedPtr> point_buf;
//queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<nav_msgs::msg::Odometry::ConstSharedPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;
double MIN_SCORE = 0.015;
double PNP_INFLATION = 1.0;
int RECALL_IGNORE_RECENT_COUNT = 50;
double MAX_THETA_DIFF = 30.0;
double MAX_POS_DIFF = 20.0;
int MIN_LOOP_NUM = 25;
bool SAVE_CAM_POSES;
double MIN_OPTIMIZATION_TIME_DIFF;
const double TIMESTAMP_SYNC_TOLERANCE = 0.000001; // seconds

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr m_camera;
double max_focallength = 460.0;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string POSE_GRAPH_LOAD_PATH;
std::string LOOP_RESULT_FOLDER;
std::string VINS_ALL_TRAJ_FOLDER;
std::string VINS_RESULT_FOLDER;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;
double image_stream_start_time = -1;
bool pose_graph_auto_saved = false;
bool has_image_arrival = false;
std::chrono::steady_clock::time_point last_image_arrival;
rclcpp::TimerBase::SharedPtr image_inactive_timer;


void new_sequence()
{
    printf("[POSEGRAPH]: new sequence\n");
    sequence++;
    printf("[POSEGRAPH]: sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        //ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        //ROS_BREAK();
        RCLCPP_WARN(rclcpp::get_logger("loop_fusion"),
                    "Only support 5 sequences; extra sequences will be ignored.");
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
    image_stream_start_time = -1;
    pose_graph_auto_saved = false;
}

bool App::ensure_dir(const std::string& path)
{
    try
    {
        if (std::filesystem::exists(path))
            return std::filesystem::is_directory(path);

        return std::filesystem::create_directories(path);
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Failed to create directory: "
                  << path << "\n"
                  << e.what() << std::endl;
        return false;
    }
}

void App::delete_files(const std::string& path)
{
    try
    {
        if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path))
            return;

        for (const auto& entry : std::filesystem::directory_iterator(path))
        {
            if (std::filesystem::is_directory(entry.status()))
            {
                delete_files(entry.path().string());
            }
            else if (std::filesystem::is_regular_file(entry.status()))
            {
                std::filesystem::remove(entry.path());
            }
            // symlinks / others are ignored
        }
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Filesystem error: "
                  << e.what() << std::endl;
    }
}

void App::image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf("[POSEGRAPH]:  image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream
    const double image_time = rclcpp::Time(image_msg->header.stamp).seconds();
    if (last_image_time < 0.0)
        last_image_time = image_time;
    else if (image_time - last_image_time > 1.0 || image_time < last_image_time)
    {
        //ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_time;

    if (image_stream_start_time < 0.0)
        image_stream_start_time = image_time;

    // Use wall time for inactivity checks to avoid ROS/sim time mismatches.
    last_image_arrival = std::chrono::steady_clock::now();
    has_image_arrival = true;
    if (pose_graph_auto_saved)
        pose_graph_auto_saved = false;
}

void App::point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg)
{
    //ROS_INFO("point_callback!");
    m_buf.lock();
    point_buf.push(point_msg); // In ROS1 it was just "push"
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("[POSEGRAPH]: %d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
    // for visualization
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);
}

// only for visualization
void App::margin_point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg)
{
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_margin_cloud->publish(point_cloud);
}

void App::pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
    /*
    printf("[POSEGRAPH]: pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void App::vio_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    nav_msgs::msg::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "global";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    odometry.twist = pose_msg->twist;
    odometry.pose.covariance = pose_msg->pose.covariance;
    pub_odometry_rect->publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);


}

void App::vio_callback_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    geometry_msgs::msg::PoseWithCovarianceStamped odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "global";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    odometry.pose.covariance = pose_msg->pose.covariance;
    pub_pose_rect->publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);


}

void App::extrinsic_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void App::intrinsics_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    m_process.lock();
    assert(msg->k.size()==9);
    assert(msg->k.size()==4);
    cv::Size imageSize(msg->width, msg->height);
    if(msg->distortion_model == "plumb_bob") {
        m_camera = camodocal::CameraFactory::instance()->generateCamera(camodocal::Camera::ModelType::PINHOLE, "cam0", imageSize);
        std::vector<double> parameters;
        parameters.push_back(msg->d.at(0));
        parameters.push_back(msg->d.at(1));
        parameters.push_back(msg->d.at(2));
        parameters.push_back(msg->d.at(3));
        parameters.push_back(msg->k.at(0));
        parameters.push_back(msg->k.at(4));
        parameters.push_back(msg->k.at(2));
        parameters.push_back(msg->k.at(5));
        m_camera.get()->readParameters(parameters);
        max_focallength = std::max(msg->k.at(0), msg->k.at(4));
    } else if(msg->distortion_model == "equidistant") {
        m_camera = camodocal::CameraFactory::instance()->generateCamera(camodocal::Camera::ModelType::KANNALA_BRANDT, "cam0", imageSize);
        std::vector<double> parameters;
        parameters.push_back(msg->d.at(0));
        parameters.push_back(msg->d.at(1));
        parameters.push_back(msg->d.at(2));
        parameters.push_back(msg->d.at(3));
        parameters.push_back(msg->k.at(0));
        parameters.push_back(msg->k.at(4));
        parameters.push_back(msg->k.at(2));
        parameters.push_back(msg->k.at(5));
        m_camera.get()->readParameters(parameters);
        max_focallength = std::max(msg->k.at(0), msg->k.at(4));
    } else {
        throw std::runtime_error("Invalid distorition model, unable to parse (plumb_bob, equidistant)");
    }
    m_process.unlock();
}

void process()
{
    while (true)
    {
        sensor_msgs::msg::Image::ConstSharedPtr image_msg = NULL;
        sensor_msgs::msg::PointCloud::ConstSharedPtr point_msg = NULL;
        nav_msgs::msg::Odometry::ConstSharedPtr pose_msg = NULL;

        auto stamp_seconds = [](const rclcpp::Time &t) { return t.seconds(); };

        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            const double image_front_time = stamp_seconds(rclcpp::Time(image_buf.front()->header.stamp));
            const double pose_front_time  = stamp_seconds(rclcpp::Time(pose_buf.front()->header.stamp));
            const double point_front_time = stamp_seconds(rclcpp::Time(point_buf.front()->header.stamp));

            if (image_front_time - pose_front_time > TIMESTAMP_SYNC_TOLERANCE)
            {
                pose_buf.pop();
                printf("[POSEGRAPH]: throw pose at beginning | image: %.9f  pose: %.9f\n", image_front_time, pose_front_time);
            }
            else if (image_front_time - point_front_time > TIMESTAMP_SYNC_TOLERANCE)
            {
                point_buf.pop();
                printf("[POSEGRAPH]: throw point at beginning | image: %.9f  point: %.9f\n", image_front_time, point_front_time);
            }
            else if (stamp_seconds(rclcpp::Time(image_buf.back()->header.stamp)) + TIMESTAMP_SYNC_TOLERANCE >= pose_front_time
                     && stamp_seconds(rclcpp::Time(point_buf.back()->header.stamp)) + TIMESTAMP_SYNC_TOLERANCE >= pose_front_time)
            {
                // printf("[POSEGRAPH]: sync ok | image(front/back): %.9f/%.9f  point(front/back): %.9f/%.9f  pose(front): %.9f  tol: %.3fms\n",
                //        image_front_time,
                //        stamp_seconds(rclcpp::Time(image_buf.back()->header.stamp)),
                //        point_front_time,
                //        stamp_seconds(rclcpp::Time(point_buf.back()->header.stamp)),
                //        pose_front_time,
                //        TIMESTAMP_SYNC_TOLERANCE * 1000.0);
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (stamp_seconds(rclcpp::Time(image_buf.front()->header.stamp)) + TIMESTAMP_SYNC_TOLERANCE < pose_front_time)
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (stamp_seconds(rclcpp::Time(point_buf.front()->header.stamp)) + TIMESTAMP_SYNC_TOLERANCE < pose_front_time)
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            //printf("[POSEGRAPH]:  pose time %f \n", pose_msg->header.stamp.toSec());
            //printf("[POSEGRAPH]:  point time %f \n", point_msg->header.stamp.toSec());
            //printf("[POSEGRAPH]:  image time %f \n", image_msg->header.stamp.toSec());
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::msg::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            //cv::equalizeHist(image, image);

            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    //printf("[POSEGRAPH]: u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }

                const double pose_time = rclcpp::Time(pose_msg->header.stamp).seconds();
                KeyFrame* keyframe = new KeyFrame(pose_time, frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("[POSEGRAPH]: save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("[POSEGRAPH]: program shutting down...\n");
            rclcpp::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

// Note: ROS2 throws an error if the same parameter is declared twice
template <class T>
void declareParameter(rclcpp::Node::SharedPtr nh, const std::string &param_field, const T &type) {
  try {
    nh->declare_parameter(param_field, type);
  } catch (rclcpp::exceptions::ParameterAlreadyDeclaredException &e) {
    std::cerr << "Error declaring parameter: " << e.what() << "\n";
  }
}

void getParamOrExit(rclcpp::Node::SharedPtr nh, const std::string &param_field, std::string& variable){
  //std::cout << param_field << " is of string\n" << std::flush;
  declareParameter(nh, param_field, rclcpp::PARAMETER_STRING);
  if (!nh->get_parameter(param_field, variable)) {
    throw std::invalid_argument("Exiting. Couldn't find param: " + param_field);
  }
  std::cout << param_field << ": " << variable <<" (string)\n";
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv); // ros::init_options::AnonymousName);
    rclcpp::Node::SharedPtr nh = rclcpp::Node::make_shared("loop_fusion");
  
    CommandLineConfig app_params;
    std::shared_ptr<App> app = std::make_shared<App>(nh, app_params);
    posegraph.registerPub(nh);

    auto save_topic = nh->create_subscription<std_msgs::msg::Bool>(
        "save_pose_graph",
        10,
        [](const std_msgs::msg::Bool::SharedPtr msg) {
            if (!msg->data) {
                return;
            }
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("[POSEGRAPH]: save pose graph finish\n");
        });

    image_inactive_timer = nh->create_wall_timer(
        std::chrono::milliseconds(200),
        []() {
            if (!has_image_arrival || pose_graph_auto_saved)
                return;
            const auto now = std::chrono::steady_clock::now();
            const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_image_arrival);
            if (idle.count() >= 3000)
            {
                m_process.lock();
                posegraph.savePoseGraph();
                m_process.unlock();
                pose_graph_auto_saved = true;
                printf("[POSEGRAPH]: auto-saved pose graph after 3s of no images\n");
            }
        });
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

   
    string config_file, vocabulary_file, brief_pattern_file;
    getParamOrExit(nh, "config_file", config_file);
    getParamOrExit(nh, "vocabulary_file", vocabulary_file);
    getParamOrExit(nh, "brief_pattern_file", brief_pattern_file);
    
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    BRIEF_PATTERN_FILE = brief_pattern_file;

    posegraph.loadVocabulary(vocabulary_file);

    //ROW = fsSettings["image_height"];
    //COL = fsSettings["image_width"];
    //int pn = config_file.find_last_of('/');
    //std::string configPath = config_file.substr(0, pn);
    //std::string cam0Calib;
    //fsSettings["cam0_calib"] >> cam0Calib;
    //std::string cam0Path = configPath + "/" + cam0Calib;
    //printf("[POSEGRAPH]: cam calib path: %s\n", cam0Path.c_str());
    //m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());


    fsSettings["output_path"] >> VINS_RESULT_FOLDER;
    VINS_RESULT_PATH = VINS_RESULT_FOLDER + "/ov_secondary.txt";
    VINS_ALL_TRAJ_FOLDER = VINS_RESULT_FOLDER + "/all_traj/";
    LOOP_RESULT_FOLDER = VINS_RESULT_FOLDER + "/loops/";
    POSE_GRAPH_SAVE_PATH = VINS_RESULT_FOLDER + "/pose_graph/";
    fsSettings["pose_graph_load_path"] >> POSE_GRAPH_LOAD_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;
    fsSettings["save_cam_poses"] >> SAVE_CAM_POSES;
    fsSettings["skip_dist"] >> SKIP_DIS;
    fsSettings["skip_cnt"] >> SKIP_CNT;
    fsSettings["min_score"] >> MIN_SCORE;
    fsSettings["pnp_inflation"] >> PNP_INFLATION;
    fsSettings["recall_ignore_recent_ct"] >> RECALL_IGNORE_RECENT_COUNT;
    fsSettings["max_theta_diff"] >> MAX_THETA_DIFF;
    fsSettings["max_pos_diff"] >> MAX_POS_DIFF;
    fsSettings["min_loop_feat_num"] >> MIN_LOOP_NUM;
    fsSettings["min_optimization_time_diff"] >> MIN_OPTIMIZATION_TIME_DIFF;

    if (!std::filesystem::exists(VINS_RESULT_FOLDER) || !std::filesystem::is_directory(VINS_RESULT_FOLDER))
    {
        throw std::runtime_error(
            "VINS_RESULT_FOLDER does not exist or is not a directory: " +
            VINS_RESULT_FOLDER);
    }

    app->ensure_dir(VINS_ALL_TRAJ_FOLDER);
    app->ensure_dir(LOOP_RESULT_FOLDER);
    app->ensure_dir(POSE_GRAPH_SAVE_PATH);

    app->delete_files(VINS_RESULT_FOLDER);

    int LOAD_PREVIOUS_POSE_GRAPH;
    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
    std::ofstream loops_out(LOOP_RESULT_FOLDER + "/ov_loops.txt", std::ios::out);
    loops_out.close();

    int USE_IMU = fsSettings["imu"];
    posegraph.setIMUFlag(USE_IMU);
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("[POSEGRAPH]: load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("[POSEGRAPH]: load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("[POSEGRAPH]: no previous pose graph\n");
        load_flag = 1;
    }

    /*
    // Get camera information
    printf("[POSEGRAPH]: waiting for camera info topic...\n");
    auto msg1 = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/vins_estimator/intrinsics", ros::Duration(ros::DURATION_MAX));
    intrinsics_callback(msg1);
    printf("[POSEGRAPH]: received camera info message!\n");
    std::cout << m_camera.get()->parametersToString() << std::endl;

    // Get camera to imu information
    printf("[POSEGRAPH]: waiting for camera to imu extrinsics topic...\n");
    auto msg2 = ros::topic::waitForMessage<nav_msgs::Odometry>("/vins_estimator/extrinsic", ros::Duration(ros::DURATION_MAX));
    extrinsic_callback(msg2);
    printf("[POSEGRAPH]: received camera to imu extrinsics message!\n");
    std::cout << qic.transpose() << std::endl;
    std::cout << tic.transpose() << std::endl;
    */

    // Get camera information
    printf("[POSEGRAPH]: waiting for camera info topic...\n");
    sensor_msgs::msg::CameraInfo msg1;
    rclcpp::wait_for_message(msg1, nh, "/ov_msckf/loop_intrinsics");
    app->intrinsics_callback(std::make_shared<sensor_msgs::msg::CameraInfo>(msg1));
    printf("[POSEGRAPH]: received camera info message!\n");
    std::cout << m_camera.get()->parametersToString() << std::endl;

    // Get camera to imu information
    printf("[POSEGRAPH]: waiting for camera to imu extrinsics topic...\n");
    nav_msgs::msg::Odometry msg2;
    rclcpp::wait_for_message(msg2, nh, "/ov_msckf/loop_extrinsic");
    app->extrinsic_callback(std::make_shared<nav_msgs::msg::Odometry>(msg2));
    printf("[POSEGRAPH]: received camera to imu extrinsics message!\n");
    std::cout << qic.transpose() << std::endl;
    std::cout << tic.transpose() << std::endl;

    // Setup the rest of the publishers
    // auto sub_vio1 = nh->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/odometry", 2000, std::bind(&App::vio_callback, app.get(), std::placeholders::_1));
    auto sub_vio2 = nh->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/ov_msckf/poseimu", 2000, std::bind(&App::vio_callback_pose, app.get(), std::placeholders::_1));
    auto sub_image = nh->create_subscription<sensor_msgs::msg::Image>("/cam0/image_raw", 2000, std::bind(&App::image_callback, app.get(), std::placeholders::_1));
    auto sub_pose = nh->create_subscription<nav_msgs::msg::Odometry>("/ov_msckf/loop_pose", 2000, std::bind(&App::pose_callback, app.get(), std::placeholders::_1));
    auto sub_extrinsic = nh->create_subscription<nav_msgs::msg::Odometry>("/ov_msckf/loop_extrinsic", 2000, std::bind(&App::extrinsic_callback, app.get(), std::placeholders::_1));
    auto sub_intrinsics = nh->create_subscription<sensor_msgs::msg::CameraInfo>("/ov_msckf/loop_intrinsics", 2000, std::bind(&App::intrinsics_callback, app.get(), std::placeholders::_1));
    auto sub_point = nh->create_subscription<sensor_msgs::msg::PointCloud>("/ov_msckf/loop_feats", 2000, std::bind(&App::point_callback, app.get(), std::placeholders::_1));
    // auto sub_margin_point = nh->create_subscription<sensor_msgs::msg::PointCloud>("/vins_estimator/margin_cloud", 2000, std::bind(&App::margin_point_callback, app.get(), std::placeholders::_1));

    std::thread measurement_process;
    std::thread keyboard_command_process;

    measurement_process = std::thread(process);
    keyboard_command_process = std::thread(command);
    
    RCLCPP_INFO_STREAM(nh->get_logger(), "pose_graph_node ready");
    rclcpp::spin(nh);

    return 0;
}
