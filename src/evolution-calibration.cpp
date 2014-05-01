#include <cstdlib>

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <Velodyne.h>
#include <Calibration.h>
#include <Image.h>
#include <Similarity.h>

#include <opencv/cv.h>

using namespace std;

static mt19937 GENERATOR;

class CalibrationSubspace
{
public:
  CalibrationSubspace(Calibration6DoF initial, float distance_translation, float distance_rotation,
                      SimilarityCameraLidar similarity) :
      similarity(similarity), DIST_TRANSL(distance_translation), DIST_ROT(distance_rotation)
  {
    for (int i = 0; i < DOF; i++)
    {
      float distance = (i < DOF / 2) ? distance_translation : distance_rotation;
      range_min.push_back(initial.DoF[i] - distance);
      range_max.push_back(initial.DoF[i] + distance);
    }
  }

  float perturb(Calibration6DoF original, int index, float sigma)
  {
    normal_distribution<> gauss(0, sigma);
    float new_value = original.DoF[index] + gauss(GENERATOR);

    // reflection:
    while (new_value > range_max[index] || new_value < range_min[index])
    {
      if (new_value > range_max[index])
      {
        new_value = 2 * range_max[index] - new_value;
      }
      else
      {
        new_value = 2 * range_min[index] - new_value;
      }
    }
    return new_value;
  }

  Calibration6DoF perturb(Calibration6DoF original, float sigma)
  {
    Calibration6DoF perturbed;
    for (int i = 0; i < DOF; i++)
    {
      perturbed.DoF[i] = perturb(original, i, sigma);
    }
    perturbed.value = evaluate(perturbed);
    return perturbed;
  }

  Calibration6DoF genRandom()
  {
    Calibration6DoF generated;
    generated.DoF.clear();
    for (int i = 0; i < DOF; i++)
    {
      uniform_real_distribution<double> uniform(range_min[i], range_max[i]);
      generated.DoF.push_back(uniform(GENERATOR));
    }
    generated.value = evaluate(generated);
    return generated;
  }

  float evaluate(Calibration6DoF calibration)
  {
    return similarity.calibrationValue(calibration.DoF);
  }

protected:
  vector<float> range_min, range_max;
  SimilarityCameraLidar similarity;
public:
  static const int DOF = 6;
  const float DIST_TRANSL, DIST_ROT;
};

Calibration6DoF evolution1x1(Calibration6DoF initial, CalibrationSubspace subspace, float sigma)
{
  Calibration6DoF x, x_;
  x = initial;
  float c_d = 0.82;
  float c_i = 1 / c_d;
  int attempts = 10;

  // 1min timeout
  int counter = 0;
  double t_max = cv::getTickCount() + 60 * cv::getTickFrequency();
  while (x.value < 0.9168 && cv::getTickCount() < t_max)
  {
    int successes = 0;
    for (int i = 0; i < attempts; i++)
    {
      x_ = subspace.perturb(x, sigma);
      if (x_.value > x.value)
      {
        successes++;
        x = x_;
        //x.print();
      }
      counter++;
    }
  }
  cerr << counter << "vectors searched" << endl;
  return x;
}

class Member
{
public:
  // generating new member of the population:
  Member(CalibrationSubspace &subspace, float delta_sigma) :
      DELTA_SIGMA(delta_sigma)
  {
    calibration = subspace.genRandom();
    for (int i = 0; i < CalibrationSubspace::DOF; i++)
    {
      sigmas.push_back((i < CalibrationSubspace::DOF / 2) ? subspace.DIST_TRANSL : subspace.DIST_ROT);
    }
  }

  // crossing:
  Member(Member &p1, Member &p2, CalibrationSubspace &subspace) :
      DELTA_SIGMA(p1.DELTA_SIGMA)
  {
    calibration.DoF.clear();
    int rnd = rand();
    for (int i = 0; i < CalibrationSubspace::DOF; i++)
    {
      if (rnd & 1)
        calibration.DoF.push_back(p1.calibration.DoF[i]);
      else
        calibration.DoF.push_back(p2.calibration.DoF[i]);
      rnd >>= 1;

      if (rnd & 1)
        sigmas.push_back(p1.sigmas[i]);
      else
        sigmas.push_back(p2.sigmas[i]);
      rnd >>= 1;
    }
    perturb(sigmas, subspace);
    sigmasMutation();
    calibration.value = subspace.evaluate(calibration);
  }

  // from better to worst calibrations
  bool operator <(const Member& other) const
  {
    return this->calibration.value > other.calibration.value;
  }

  static void generate(int size, vector<Member> &population, CalibrationSubspace &subspace, float delta_sigma)
  {
    for (int i = 0; i < size; i++)
    {
      population.push_back(Member(subspace, delta_sigma));
    }
  }

  void print()
  {
    calibration.print();
    cout << "sigmas: ";
    for (int i = 0; i < CalibrationSubspace::DOF; i++)
    {
      cout << sigmas[i] << " ";
    }
    cout << endl;
  }

protected:
  void perturb(vector<float> sigmas, CalibrationSubspace &subspace)
  {
    assert(sigmas.size() == CalibrationSubspace::DOF);
    Calibration6DoF perturbed;
    for (int i = 0; i < CalibrationSubspace::DOF; i++)
    {
      normal_distribution<> gauss(0, sigmas[i]);
      perturbed.DoF[i] = subspace.perturb(calibration, i, sigmas[i]);
    }
  }
  void sigmasMutation()
  {
    normal_distribution<> gauss(0, DELTA_SIGMA);
    for (int i = 0; i < CalibrationSubspace::DOF; i++)
    {
      sigmas[i] *= std::exp(gauss(GENERATOR));
    }
  }
public:
  Calibration6DoF calibration;
  vector<float> sigmas;
  float DELTA_SIGMA;
};

class MultimemberEvolutionSpace
{
public:
  MultimemberEvolutionSpace(CalibrationSubspace &subspace, float transl_sigma, float rot_sigma) :
      subspace(subspace)
  {
  }

  Calibration6DoF evolutionMxN(Calibration6DoF initial, int m, int n, float delta_sigma)
  {
    Calibration6DoF best = initial;
    vector<Member> population;
    Member::generate(m, population, subspace, delta_sigma);
    sort(population.begin(), population.end());

    // 1min timeout
    int counter = 0;
    double t_max = cv::getTickCount() + 60 * cv::getTickFrequency();
    while (best.value < 0.917 && cv::getTickCount() < t_max)
    {
      if (best.value < population.front().calibration.value)
      {
        best = population.front().calibration;
        //cout << endl;
        //population.front().print();
      }
      while (population.size() < n)
      {
        int i = rand() % m;
        int j = rand() % m;
        population.push_back(Member(population[i], population[j], subspace));
      }
      //cerr << ".";
      sort(population.begin(), population.end());
      population.erase(population.begin() + m, population.end());

      counter += n;
    }
    cerr << counter << " vectors searched" << endl;
    return best;
  }

protected:
  CalibrationSubspace subspace;
};

//#define MULTIMEMBER_EVOLUTION

int main(int argc, char** argv)
{
  CalibrationInputs input = Calibration::loadArgumets(argc, argv, true);
  Image::Image img(input.frame_gray);
  Velodyne::Velodyne scan(input.pc);

  /*  pcl::PointCloud<Velodyne::Point> visible_points;
   scan.project(input.P, cv::Rect(0,0,img.size().width, img.size().height), &visible_points);
   scan = Velodyne::Velodyne(visible_points);
   scan = scan.sample(10);

   scan.save("samples_visible_plain_a.pcd");*/

  SimilarityCameraLidar similarity(img, scan, input.P);

  float distance_transl = 0.02;
  float distance_rot = 0.01;

  GENERATOR.seed(time(NULL));
  srand(time(NULL));
  Calibration6DoF initial(
      input.x, input.y, input.z, input.rot_x, input.rot_y, input.rot_z,
      similarity.calibrationValue(input.x, input.y, input.z, input.rot_x, input.rot_y, input.rot_z));
  CalibrationSubspace subspace(initial, distance_transl, distance_rot, similarity);
  Calibration6DoF best;

#ifdef MULTIMEMBER_EVOLUTION
  MultimemberEvolutionSpace multispace(subspace, distance_transl/2, distance_rot/2);
  best = multispace.evolutionMxN(initial, 100, 700, 0.03);
#else
  best = evolution1x1(initial, subspace, distance_transl / 2);
#endif

  best.print();
  return EXIT_SUCCESS;
}
