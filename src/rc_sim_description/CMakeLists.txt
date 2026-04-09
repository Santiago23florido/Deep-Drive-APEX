cmake_minimum_required(VERSION 3.8)
project(rc_sim_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

install(
  DIRECTORY launch urdf worlds config meshes
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING
    PATTERN "*.launch.py"
    PATTERN "*.json"
    PATTERN "*.yaml"
    PATTERN "*.xacro"
    PATTERN "*.rviz"
    PATTERN "*.world"
    PATTERN "*.stl"
    PATTERN "*.dae"
    PATTERN "*.obj"
)

install(
  PROGRAMS
    scripts/apex_gz_vehicle_bridge.py
    scripts/apex_ground_truth_node.py
    scripts/apex_refined_sensorfusion_map_node.py
    scripts/apex_xbox_manual_teleop_node.py
    scripts/apex_windows_gamepad_bridge_node.py
    scripts/apex_offline_sensorfusion_map_publisher.py
    scripts/apex_general_track_mapper.py
    scripts/apex_general_track_map_publisher.py
    scripts/rear_wheel_speed_publisher.py
    scripts/turning_command_mapper.py
    scripts/control_system/voiture_control_node.py
    scripts/voiture_system/rear_wheel_speed_bridge.py
    scripts/voiture_system/gazebo_lidar_reader_node.py
    scripts/algorithms/sim_lidar_reader.py
    scripts/algorithms/test_sim_lidar_reader.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY scripts/algorithms/
  DESTINATION lib/${PROJECT_NAME}/algorithms
  FILES_MATCHING
    PATTERN "*.py"
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
