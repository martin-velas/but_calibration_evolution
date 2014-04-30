#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <algorithm>

#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <Velodyne.h>
#include <Calibration.h>
#include <Image.h>

using namespace cv;
using namespace std;
using namespace pcl;

int main(int argc, char** argv)
{
  CalibrationInputs input = Calibration::loadArgumets(argc, argv, true);

  Calibration6DoF best, avg;
  size_t divisions = 5;
  float distance_transl = 0.02;
  float distance_rot = 0.01;
  Calibration::calibrationRefinement(Image::Image(input.frame_gray), Velodyne::Velodyne(input.pc), input.P,
                                     input.x, input.y, input.z, distance_transl, distance_rot, divisions,
                                     best, avg);
  best.print();

  return EXIT_SUCCESS;
}
