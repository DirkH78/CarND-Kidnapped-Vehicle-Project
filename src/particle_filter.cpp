/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// no. of particles
  // check initialization
  if (is_initialized) {
    return;
  }
  else {
    // number of particles ---this is a variable to be adapted---
    num_particles = 30;
    //generator
    static default_random_engine gen;
  
    // This line creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
  
    // initialize particles with normal distributed initialization values
    for (int i = 0; i < num_particles; i++){
      Particle p;
      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0;
    
      particles.push_back(p);
    }
  // initialization finished
  is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  //generator
  static default_random_engine gen;
  
  // use equations of motion for yaw rate == =0 and yaw rate <> 0
  for (int i = 0; i < num_particles; i++){
    if (fabs(yaw_rate) < 0.0001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // This line creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i++) {
    // set largest possible value as start value for actual_minimum
    double actual_minimum = numeric_limits<double>::max();
    int actual_id = 0;
    for (int n = 0; n < predicted.size(); n++) {
      // calculate geomatrical distance between all observations and predictions --- beware of high computational effort (n*i)! 
      double distance = sqrt(pow((observations[i].x - predicted[n].x), 2) + pow((observations[i].y - predicted[n].y), 2));
      // set smallest distance to be "nearest neighbor"
      if (distance < actual_minimum){
        actual_minimum = distance;
        actual_id = predicted[n].id;
      }
    }
    observations[i].id = actual_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  // only take into account landmarks within range of sensor to limit computational effort [see jeremy-shannon]
  for (int i = 0; i < num_particles; i++) {
    vector<LandmarkObs> predictions;
    for (int n = 0; n < map_landmarks.landmark_list.size(); n++) {
      if (fabs(map_landmarks.landmark_list[n].x_f - particles[i].x) <= sensor_range && fabs(map_landmarks.landmark_list[n].y_f - particles[i].y) <= sensor_range) {
        predictions.push_back(LandmarkObs{map_landmarks.landmark_list[n].id_i, map_landmarks.landmark_list[n].x_f, map_landmarks.landmark_list[n].y_f});
      }
    }
    // perform coordinate transformation save in "transformed"
    vector<LandmarkObs> transformed;
    for (int m = 0; m < observations.size(); m++) {
      double trans_x = cos(particles[i].theta) * observations[m].x - sin(particles[i].theta) * observations[m].y + particles[i].x;
      double trans_y = sin(particles[i].theta) * observations[m].x + cos(particles[i].theta) * observations[m].y + particles[i].y;
      transformed.push_back(LandmarkObs{observations[m].id, trans_x, trans_y});
    }
    // use dataAssociation to identify nearest neighbors
    dataAssociation(predictions, transformed);
    particles[i].weight = 1.0;
    for (int o = 0; o < transformed.size(); o++) {
      double pred_x, pred_y;
      for (int p = 0; p < predictions.size(); p++) {
        // only use nearest neighbors
        if (predictions[p].id == transformed[o].id) {
          pred_x = predictions[p].x;
          pred_y = predictions[p].y;
        }
      }
      // calculate new weight by using a multi-variate Gaussian distribution on distances
      particles[i].weight *= (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp( -( pow(pred_x - transformed[o].x, 2) / (2 * pow(std_landmark[0], 2)) + (pow(pred_y - transformed[o].y, 2) / (2 * pow(std_landmark[1], 2)))));
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  // generator
  default_random_engine gen;
  // generate new set of particles and weights
  vector<Particle> new_set;
  vector<double> weights;
  // use the old particles weights to initialize new weights
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  // utilize a resampling wheel as shown in course (Python-->C++)
  // generate random starting index for resampling wheel
  discrete_distribution<int> init_index(weights.begin(), weights.end());
  int index = init_index(gen);
  
  // find max weight and initiate discrete distribution between 0 and max
  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> weight_dist(0.0, max_weight);
  double random_weight = weight_dist(gen);
  
  // initiate index beta with 0
  double beta = 0.0;

  // spinning the resample wheel
  for (int i = 0; i < num_particles; i++) {
    beta += random_weight * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_set.push_back(particles[index]);
  }
  particles = new_set;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
