cmake_minimum_required(VERSION 3.8)
project(query_llm_behavior)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# Non-ROS packages
find_package(OpenCV REQUIRED)

# Install header files (entire include tree)
install(DIRECTORY include/
  DESTINATION include
)

# Export to downstream packages
ament_export_include_directories(include)
ament_export_dependencies(rclcpp behaviortree_cpp sensor_msgs OpenCV)

# No library target is created — this is header-only

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
