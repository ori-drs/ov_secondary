#include <rclcpp/node.hpp>
#include <filesystem> //C++17

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

using namespace std;

struct CommandLineConfig
{
    string temp;

};

class App {
public:
    App(rclcpp::Node::SharedPtr node, const CommandLineConfig &app_params);

    ~App(){
    }
    
    
    void vio_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void vio_callback_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg);
    void image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg);
    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg);
    void extrinsic_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg);
    void intrinsics_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg);
    void margin_point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg);

    bool ensure_dir(const std::string& path);
    void delete_files(const std::string& path);


private:
    rclcpp::Node::SharedPtr node_;
    CommandLineConfig app_params_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_rect;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_rect;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_margin_cloud;
    
};

