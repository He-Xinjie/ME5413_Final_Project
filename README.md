# 快速上手

```bash
unzip 2.zip && cd 2
cp -r src/me5413_world/models/* ~/.gazebo/models/          # 一次性：gazebo 模型
rosdep install --from-paths src --ignore-src -r -y && catkin_make && source devel/setup.bash

# 终端 1
roslaunch me5413_world world.launch

# 终端 2
source devel/setup.bash
roslaunch me5413_team_solution navigation.launch
```
