cmake_minimum_required(VERSION 3.8)
project(query_llm_behavior)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)

find_package(spot_utils REQUIRED)
# Specify include directories
include_directories(
  include
)

# Add libraries
add_library(${PROJECT_NAME} src/${PROJECT_NAME}/${PROJECT_NAME}.cpp)

# Specify ROS dependencies for the libraries
ament_target_dependencies(query_llm rclcpp behaviortree_cpp spot_utils)

# Install the 'include' directory
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Install the libraries
install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
