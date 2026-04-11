/* rviz_panel.hpp
 * RViz Panel for ME5413 Navigation Control
 * 3 zone buttons: Lower Level / Stairway / Upper Level
 * Waypoints saved to separate files per zone
 */

#ifndef rviz_panel_H_
#define rviz_panel_H_

#include <ros/ros.h>
#include <rviz/panel.h>
#include <ui_simple_panel.h>

#include <std_msgs/Int16.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>

#include <string>
#include <map>

namespace rviz_panel
{
class ME5413ControlPanel : public rviz::Panel
{
  Q_OBJECT

 public:
  #ifdef UNIT_TEST
    friend class testClass;
  #endif
  ME5413ControlPanel(QWidget *parent = 0);

  virtual void save(rviz::Config config) const;
  virtual void load(const rviz::Config &config);

  public Q_SLOTS:
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
};

} // namespace rviz_panel

#endif
