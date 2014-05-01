/*
 * Similarity.h
 *
 *  Created on: 8.1.2014
 *      Author: ivelas
 */

#ifndef SIMILARITY_H_
#define SIMILARITY_H_

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <map>
#include <utility>
#include <cassert>
#include <string>
#include <sstream>

#include <ros/assert.h>

#include <Exceptions.h>
#include <Image.h>
#include <Velodyne.h>

class Similarity2D
{
public:
  Similarity2D(cv::Mat _X, cv::Mat _Y) :
      X(_X), Y(_Y), H_X(-1.0), H_Y(-1.0), H_XY(-1.0)
  {
    assert(X.type() == CV_8UC1);
    assert(Y.type() == CV_8UC1);
    assert(X.size() == Y.size());
  }

  // cross-corelation
  float getCrossCorelation()
  {
    return sum(X.mul(Y))[0];
  }

  // mutual information
  float getMutualInformation()
  {
    if (H_X < 0)
    {
      computeEntropies();
    }
    return H_X + H_Y - H_XY;
  }

  // normalized mutual information
  float getNormalizedMutualInformation()
  {
    if (H_X < 0)
    {
      computeEntropies();
    }
    return (H_X + H_Y) / H_XY;
  }

  enum Criteria
  {
    MI, NMI, CC
  };

  static Criteria getCriteria(std::string s)
  {
    if (s == "MI")
      return MI;
    if (s == "NMI")
      return NMI;
    if (s == "CC")
      return CC;
    throw new NotImplementedException("Unknown criteria " + s);
  }

  float getSimilarity(Criteria crit)
  {
    switch (crit)
    {
      case MI:
        return getMutualInformation();
      case NMI:
        return getNormalizedMutualInformation();
      case CC:
        return getCrossCorelation();
      default:
        std::stringstream ss;
        ss << "Similarity criteria " << crit << ".";
        throw new NotImplementedException(ss.str());
    }
  }
protected:
  void computeEntropies();

protected:
  cv::Mat X, Y; // compared images

  float H_X, H_Y; // entropy of image X, Y
  float H_XY; // joint entropy

  static const int INTENSITIES = 256;
};

class SimilarityCameraLidar
{
public:
  typedef enum
  {
    EDGE_SIMILARITY, PROJECTION_RATIO
  } SimilarityMethod;

  SimilarityCameraLidar(Image::Image &img, Velodyne::Velodyne &scan, cv::Mat &P,
                        SimilarityMethod method = SimilarityMethod::PROJECTION_RATIO) :
      img(img), method(method), P(P)
  {
    this->scan = this->original_scan = scan;

    img_segments = this->img.segmentation(2);
    scan_segments = original_scan_segments = this->scan.depthSegmentation(2);
  }

  float projectionError(bool verbose = false);
  float edgeSimilarity();

  float calibrationValue(std::vector<float> DoF, bool verbose = false)
  {
    assert(DoF.size() == 6);
    return calibrationValue(DoF[0], DoF[1], DoF[2], DoF[3], DoF[4], DoF[5], verbose);
  }

  float calibrationValue(float x, float y, float z, float rx, float ry, float rz, bool verbose = false)
  {
    transform(x, y, z, rx, ry, rz);
    if (method == PROJECTION_RATIO)
    {
      return (1 - projectionError(verbose));
    }
    else if (method == EDGE_SIMILARITY)
    {
      return edgeSimilarity();
    }
    else
    {
      throw new NotImplementedException("Unknown camera-lidar similarity type" + method);
    }
  }

protected:
  void transform(float x, float y, float z, float rx, float ry, float rz)
  {
    for (int i = 0; i < original_scan_segments.size(); i++)
    {
      scan_segments[i] = original_scan_segments[i].transform(x, y, z, rx, ry, rz);
    }
    scan = original_scan.transform(x, y, z, rx, ry, rz);
  }

  Image::Image img;
  Velodyne::Velodyne scan;
  Velodyne::Velodyne original_scan;
  cv::Mat P;

  cv::Mat img_segments;
  std::vector<Velodyne::Velodyne> scan_segments;
  std::vector<Velodyne::Velodyne> original_scan_segments;

  SimilarityMethod method;

};

#endif /* SIMILARITY_H_ */
