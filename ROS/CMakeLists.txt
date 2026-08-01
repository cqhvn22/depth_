cmake_minimum_required(VERSION 3.8)
project(stereo_camera_ros2)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(camera_calibration_parsers REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core videoio)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

add_library(stereo_camera_driver STATIC
  src/stereo_camera.cpp
)
target_compile_features(stereo_camera_driver PUBLIC cxx_std_17)
target_include_directories(stereo_camera_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(stereo_camera_driver PUBLIC ${OpenCV_LIBRARIES} Threads::Threads)

add_executable(stereo_camera_node
  src/stereo_camera_node.cpp
)
target_compile_features(stereo_camera_node PRIVATE cxx_std_17)
target_include_directories(stereo_camera_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(stereo_camera_node
  ament_index_cpp
  camera_calibration_parsers
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(stereo_camera_node
  stereo_camera_driver
  ${OpenCV_LIBRARIES}
)

add_library(icm20948_driver STATIC src/ICM20948.c)
target_include_directories(icm20948_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(icm20948_driver PRIVATE c_std_99)
target_link_libraries(icm20948_driver m)

add_executable(icm20948_node src/icm20948_node.cpp)
target_compile_features(icm20948_node PRIVATE cxx_std_17)
target_include_directories(icm20948_node PRIVATE include)
ament_target_dependencies(icm20948_node rclcpp sensor_msgs)
target_link_libraries(icm20948_node icm20948_driver m)

install(TARGETS
  stereo_camera_driver
  icm20948_driver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(TARGETS
  stereo_camera_node
  icm20948_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS examples/python_stereo_subscriber.py
  DESTINATION share/${PROJECT_NAME}/examples)

ament_export_include_directories(include)
ament_package()
