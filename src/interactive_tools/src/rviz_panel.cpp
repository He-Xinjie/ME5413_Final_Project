/* rviz_panel.cpp
 * RViz Panel for ME5413 Navigation Control
 * Publishes zone commands to /nav_zone/command
 * Saves 2D Nav Goal waypoints to per-zone YAML files
 */

#include <pluginlib/class_list_macros.hpp>
#include "interactive_tools/rviz_panel.hpp"
#include <ros/package.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <vector>
#include <string>

PLUGINLIB_EXPORT_CLASS(rviz_panel::ME5413ControlPanel, rviz::Panel)

namespace rviz_panel
{
    ME5413ControlPanel::ME5413ControlPanel(QWidget * parent)
    :   rviz::Panel(parent),
        ui_(std::make_shared<Ui::TaskControlPanel>())
    {
        ui_->setupUi(this);

        // Publishers
        this->pub_goal_ = nh_.advertise<std_msgs::String>("/rviz_panel/goal_name", 1);
        this->pub_respawn_ = nh_.advertise<std_msgs::Int16>("/rviz_panel/respawn_objects", 1);
        this->pub_zone_cmd_ = nh_.advertise<std_msgs::String>("/nav_zone/command", 1);

        // Connect buttons
        connect(ui_->pushButton_regen, SIGNAL(clicked()), this, SLOT(on_button_regen_clicked()));
        connect(ui_->pushButton_clear, SIGNAL(clicked()), this, SLOT(on_button_clear_clicked()));
        connect(ui_->pushButton_lower, SIGNAL(clicked()), this, SLOT(on_button_lower_clicked()));
        connect(ui_->pushButton_stairway, SIGNAL(clicked()), this, SLOT(on_button_stairway_clicked()));
        connect(ui_->pushButton_upper, SIGNAL(clicked()), this, SLOT(on_button_upper_clicked()));
        connect(ui_->pushButton_stop, SIGNAL(clicked()), this, SLOT(on_button_stop_clicked()));
        connect(ui_->pushButton_save_wp, SIGNAL(clicked()), this, SLOT(on_button_save_wp_clicked()));
        connect(ui_->pushButton_del_last_wp, SIGNAL(clicked()), this, SLOT(on_button_del_last_wp_clicked()));
        connect(ui_->pushButton_clear_wp, SIGNAL(clicked()), this, SLOT(on_button_clear_wp_clicked()));

        // Subscriber for 2D Nav Goal
        this->sub_nav_goal_ = nh_.subscribe("/move_base_simple/goal", 1,
            &ME5413ControlPanel::navGoalCallback, this);

        goal_name_msg_.data = "";
        has_nav_goal_ = false;

        // Per-zone waypoint file paths
        std::string config_dir = ros::package::getPath("me5413_team_solution") + "/config";
        zone_files_[0] = config_dir + "/waypoints_lower.yaml";
        zone_files_[1] = config_dir + "/waypoints_stairway.yaml";
        zone_files_[2] = config_dir + "/waypoints_upper.yaml";

        zone_headers_[0] = "# Lower level waypoints (map frame)\n# Format: name, x, y, yaw\n";
        zone_headers_[1] = "# Stairway waypoints (map frame)\n# Format: name, x, y, yaw\n";
        zone_headers_[2] = "# Upper level waypoints (map frame)\n# Format: name, x, y, yaw\n";

        // Show initial counts on buttons
        updateButtonCounts();
    }

    // ---- Helpers ----

    std::string ME5413ControlPanel::getSelectedZoneFile() const
    {
        int idx = ui_->comboBox_zone->currentIndex();
        auto it = zone_files_.find(idx);
        if (it != zone_files_.end())
            return it->second;
        return zone_files_.at(0);
    }

    int ME5413ControlPanel::countWaypoints(const std::string& filepath) const
    {
        int count = 0;
        std::ifstream ifs(filepath);
        std::string line;
        while (std::getline(ifs, line))
        {
            if (line.find("- name:") != std::string::npos)
                count++;
        }
        return count;
    }

    void ME5413ControlPanel::updateButtonCounts()
    {
        int lower_n = countWaypoints(zone_files_.at(0));
        int stair_n = countWaypoints(zone_files_.at(1));
        int upper_n = countWaypoints(zone_files_.at(2));

        ui_->pushButton_lower->setText(
            QString("Lower (%1)").arg(lower_n));
        ui_->pushButton_stairway->setText(
            QString("Stairway (%1)").arg(stair_n));
        ui_->pushButton_upper->setText(
            QString("Upper (%1)").arg(upper_n));
    }

    // ---- Object control ----

    void ME5413ControlPanel::on_button_regen_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: Respawning Random Objects");
        ui_->label_status->setText("Respawning objects...");
        this->regen_cmd_msg_.data = 1;
        this->pub_respawn_.publish(this->regen_cmd_msg_);
    }
    void ME5413ControlPanel::on_button_clear_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: Clearing Random Objects");
        ui_->label_status->setText("Clearing objects...");
        this->regen_cmd_msg_.data = 0;
        this->pub_respawn_.publish(this->regen_cmd_msg_);
    }

    // ---- Zone navigation ----

    void ME5413ControlPanel::on_button_lower_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: Starting Lower Level Exploration");
        ui_->label_status->setText("Exploring Lower Level...");
        this->zone_cmd_msg_.data = "lower";
        this->pub_zone_cmd_.publish(this->zone_cmd_msg_);
    }
    void ME5413ControlPanel::on_button_stairway_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: Starting Stairway Navigation");
        ui_->label_status->setText("Navigating Stairway...");
        this->zone_cmd_msg_.data = "stairway";
        this->pub_zone_cmd_.publish(this->zone_cmd_msg_);
    }
    void ME5413ControlPanel::on_button_upper_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: Starting Upper Level Exploration");
        ui_->label_status->setText("Exploring Upper Level...");
        this->zone_cmd_msg_.data = "upper";
        this->pub_zone_cmd_.publish(this->zone_cmd_msg_);
    }
    void ME5413ControlPanel::on_button_stop_clicked()
    {
        ROS_INFO_STREAM("ME5413ControlPanel: STOP");
        ui_->label_status->setText("STOPPED");
        this->zone_cmd_msg_.data = "stop";
        this->pub_zone_cmd_.publish(this->zone_cmd_msg_);
    }

    // ---- Waypoint callbacks ----

    void ME5413ControlPanel::navGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        last_nav_goal_ = *msg;
        has_nav_goal_ = true;
        double x = msg->pose.position.x;
        double y = msg->pose.position.y;
        double z = msg->pose.orientation.z;
        double w = msg->pose.orientation.w;
        double yaw = std::atan2(2.0 * w * z, 1.0 - 2.0 * z * z);

        std::ostringstream ss;
        ss << std::fixed;
        ss.precision(2);
        ss << "Goal: (" << x << ", " << y << ") yaw=" << yaw;
        ui_->label_waypoint->setText(QString::fromStdString(ss.str()));
        ROS_INFO_STREAM("ME5413ControlPanel: Nav goal received: " << ss.str());
    }

    void ME5413ControlPanel::on_button_save_wp_clicked()
    {
        if (!has_nav_goal_)
        {
            ui_->label_waypoint->setText("No goal yet! Click 2D Nav Goal first.");
            return;
        }

        double x = last_nav_goal_.pose.position.x;
        double y = last_nav_goal_.pose.position.y;
        double z = last_nav_goal_.pose.orientation.z;
        double w = last_nav_goal_.pose.orientation.w;
        double yaw = std::atan2(2.0 * w * z, 1.0 - 2.0 * z * z);

        std::string filepath = getSelectedZoneFile();

        // Append to zone YAML file
        std::ofstream ofs(filepath, std::ios::app);
        if (!ofs.is_open())
        {
            ui_->label_waypoint->setText("Error: cannot write file!");
            ROS_ERROR_STREAM("Cannot open: " << filepath);
            return;
        }

        int count = countWaypoints(filepath);

        ofs << std::fixed;
        ofs.precision(4);
        ofs << "- name: wp_" << count << "\n";
        ofs << "  x: " << x << "\n";
        ofs << "  y: " << y << "\n";
        ofs << "  yaw: " << yaw << "\n";
        ofs.close();

        std::string zone_name = ui_->comboBox_zone->currentText().toStdString();
        std::ostringstream ss;
        ss << std::fixed;
        ss.precision(2);
        ss << "Saved wp_" << count << " (" << x << ", " << y << ") -> " << zone_name;
        ui_->label_waypoint->setText(QString::fromStdString(ss.str()));
        ROS_INFO_STREAM("ME5413ControlPanel: " << ss.str());

        updateButtonCounts();
    }

    void ME5413ControlPanel::on_button_del_last_wp_clicked()
    {
        std::string filepath = getSelectedZoneFile();

        // Read all lines
        std::ifstream ifs(filepath);
        if (!ifs.is_open())
        {
            ui_->label_waypoint->setText("Error: cannot read file!");
            return;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(ifs, line))
            lines.push_back(line);
        ifs.close();

        // Find the last "- name:" and remove it + its fields
        int last_entry = -1;
        for (int i = (int)lines.size() - 1; i >= 0; --i)
        {
            if (lines[i].find("- name:") != std::string::npos)
            {
                last_entry = i;
                break;
            }
        }

        if (last_entry < 0)
        {
            ui_->label_waypoint->setText("No waypoints to delete.");
            return;
        }

        // Remove from last_entry to end of its block (lines starting with "  ")
        int end = last_entry + 1;
        while (end < (int)lines.size() && lines[end].size() > 0 && lines[end][0] == ' ')
            end++;

        lines.erase(lines.begin() + last_entry, lines.begin() + end);

        // Write back
        std::ofstream ofs(filepath, std::ios::trunc);
        for (const auto& l : lines)
            ofs << l << "\n";
        ofs.close();

        std::string zone_name = ui_->comboBox_zone->currentText().toStdString();
        std::ostringstream ss;
        ss << "Deleted last WP from " << zone_name;
        ui_->label_waypoint->setText(QString::fromStdString(ss.str()));
        ROS_INFO_STREAM("ME5413ControlPanel: " << ss.str());

        updateButtonCounts();
    }

    void ME5413ControlPanel::on_button_clear_wp_clicked()
    {
        std::string filepath = getSelectedZoneFile();
        int idx = ui_->comboBox_zone->currentIndex();

        std::ofstream ofs(filepath, std::ios::trunc);
        ofs << zone_headers_.at(idx);
        ofs.close();

        std::string zone_name = ui_->comboBox_zone->currentText().toStdString();
        std::ostringstream ss;
        ss << zone_name << " waypoints cleared.";
        ui_->label_waypoint->setText(QString::fromStdString(ss.str()));
        ROS_INFO_STREAM("ME5413ControlPanel: " << ss.str());

        updateButtonCounts();
    }

    void ME5413ControlPanel::save(rviz::Config config) const
    {
        rviz::Panel::save(config);
    }

    void ME5413ControlPanel::load(const rviz::Config & config)
    {
        rviz::Panel::load(config);
    }
} // namespace rviz_panel
