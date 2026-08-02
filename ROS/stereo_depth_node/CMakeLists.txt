cmake_minimum_required(VERSION 3.8)
project(stereo_depth_node)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    src/stereo_depth_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME stereo_depth_node
)

install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
