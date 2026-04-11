/* rviz_panel.hpp
<<<<<<< HEAD

 * Copyright (C) 2023 SS47816

 * Rviz Panel for controling goal poses

**/
=======
 * RViz Panel for ME5413 Navigation Control
 * 3 zone buttons: Lower Level / Stairway / Upper Level
 * Waypoints saved to separate files per zone
 */
>>>>>>> initial commit

#ifndef rviz_panel_H_
#define rviz_panel_H_

#include <ros/ros.h>
#include <rviz/panel.h>
#include <ui_simple_panel.h>
<<<<<<< HEAD
/**
 *  Include header generated from ui file
 *  Note that you will need to use add_library function first
 *  in order to generate the header file from ui.
 */

#include <std_msgs/Int16.h>
#include <std_msgs/String.h>
=======

#include <std_msgs/Int16.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>

#include <string>
#include <map>
>>>>>>> initial commit

namespace rviz_panel
{
class ME5413ControlPanel : public rviz::Panel
{
  Q_OBJECT

 public:
  #ifdef UNIT_TEST
    friend class testClass;
  #endif
<<<<<<< HEAD
  /**
   *  QWidget subclass constructors usually take a parent widget
   *  parameter (which usually defaults to 0).  At the same time,
   *  pluginlib::ClassLoader creates instances by calling the default
   *  constructor (with no arguments). Taking the parameter and giving
   *  a default of 0 lets the default constructor work and also lets
   *  someone using the class for something else to pass in a parent
   *  widget as they normally would with Qt.
   */
  ME5413ControlPanel(QWidget *parent = 0);

  /**
   *  Now we declare overrides of rviz::Panel functions for saving and
   *  loading data from the config file.  Here the data is the topic name.
   */
=======
  ME5413ControlPanel(QWidget *parent = 0);

>>>>>>> initial commit
  virtual void save(rviz::Config config) const;
  virtual void load(const rviz::Config &config);

  public Q_SLOTS:
<<<<<<< HEAD
  /**
   *  Here we declare some internal slots.
   */
  private Q_SLOTS:
    // // Assembly Line buttons
    // void on_button_1_1_clicked();
    // void on_button_1_2_clicked();
    // // Packaging Area buttons
    // void on_button_2_1_clicked();
    // void on_button_2_2_clicked();
    // void on_button_2_3_clicked();
    // void on_button_2_4_clicked();
    // // Delivery Vehicle buttons
    // void on_button_3_1_clicked();
    // void on_button_3_2_clicked();
    // void on_button_3_3_clicked();
    // Contorl Buttons
    void on_button_regen_clicked();
    void on_button_clear_clicked();

 protected:
  // UI pointer
  std::shared_ptr<Ui::TaskControlPanel> ui_;
  // ROS declaration
  ros::NodeHandle nh_;
  ros::Publisher pub_goal_;
  ros::Publisher pub_respawn_;
  std_msgs::String goal_name_msg_;
  std_msgs::Int16 regen_cmd_msg_;
=======
  private Q_SLOTS:
    // Object control
    void on_button_regen_clicked();
    void on_button_clear_clicked();
    // Zone navigation
    void on_button_lower_clicked();
    void on_button_stairway_clicked();
    void on_button_upper_clicked();
    void on_button_stop_clicked();
    // Waypoint editing
    void on_button_save_wp_clicked();
    void on_button_del_last_wp_clicked();
    void on_button_clear_wp_clicked();
    void navGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

 protected:
  std::shared_ptr<Ui::TaskControlPanel> ui_;
  ros::NodeHandle nh_;
  ros::Publisher pub_goal_;
  ros::Publisher pub_respawn_;
  ros::Publisher pub_zone_cmd_;
  std_msgs::String goal_name_msg_;
  std_msgs::Int16 regen_cmd_msg_;
  std_msgs::String zone_cmd_msg_;

  ros::Subscriber sub_nav_goal_;
  geometry_msgs::PoseStamped last_nav_goal_;
  bool has_nav_goal_;

  // Per-zone waypoint files
  std::map<int, std::string> zone_files_;   // comboBox index -> file path
  std::map<int, std::string> zone_headers_; // comboBox index -> file header comment

  // Helpers
  std::string getSelectedZoneFile() const;
  int countWaypoints(const std::string& filepath) const;
  void updateButtonCounts();
>>>>>>> initial commit
};

} // namespace rviz_panel

<<<<<<< HEAD
#endif
=======
#endif
>>>>>>> initial commit
