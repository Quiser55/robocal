cmake_minimum_required(VERSION 3.5)
project(annin_ar4_gazebo)
find_package(ament_cmake REQUIRED)

install(DIRECTORY launch worlds config models scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
