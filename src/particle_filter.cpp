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
#include <map>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initialize the number of particles
	num_particles = 100;

	// Generator
	default_random_engine gen;

	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);


	for (int i = 0; i < num_particles; ++i) {
		// Sample  and from these normal distrubtions like this:
		//	 sample_x = dist_x(gen);
		//	 where "gen" is the random engine initialized earlier.

		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	//Setting initialization true
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	printf("Prediction: delta_t = %f  velocity = %f  yaw_rate = %f\n", delta_t, velocity, yaw_rate);

	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto& particle:particles) {

		//The equations for updating x, y and the yaw angle when the yaw rate is not equal to zero:
		//x​f​​=x​0​​+​​θ​˙​​​​v​​[sin(θ​0​​+​θ​˙​​(dt))−sin(θ​0​​)]
		//y​f​​=y​0​​+​​θ​˙​​​​v​​[cos(θ​0​​)−cos(θ​0​​+​θ​˙​​(dt))]
		//θ​f​​=θ​0​​+​θ​˙​​(dt)

		if (fabs(yaw_rate)>0.001)
		{
			particle.x += dist_x(gen) + (velocity / yaw_rate)*(sin(particle.theta + (yaw_rate*delta_t)) - sin(particle.theta));
			particle.y += dist_y(gen) + (velocity / yaw_rate)*(-cos(particle.theta + (yaw_rate*delta_t)) + cos(particle.theta));
		}
		else
		{
			particle.x += dist_x(gen) + velocity*cos(particle.theta)*delta_t;
			particle.y += dist_y(gen) + velocity*sin(particle.theta)*delta_t;
		}
		particle.theta += dist_theta(gen) + (yaw_rate*delta_t);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Nearest Neighbor Data Association
	for (int i = 0; i < observations.size(); i++) {
		double min_dist = std::numeric_limits<double>::max();

		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_dist){
				min_dist = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double std_x_2 = pow(std_x, 2);
    double std_y_2 = pow(std_y, 2);
    double std_x_y = 2 * M_PI * std_x * std_y;

    for (int i = 0; i < num_particles; i++) {
    	Particle& particle = particles[i];

        vector<LandmarkObs> predictions;
        unordered_map<int, LandmarkObs> pred_map;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        	Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
            if (fabs(particle.x - landmark.x_f) <= sensor_range && fabs(particle.y - landmark.y_f) <= sensor_range) {
                LandmarkObs pred = {landmark.id_i, landmark.x_f, landmark.y_f};
                predictions.push_back(pred);
                pred_map[landmark.id_i] = pred;
            }
        }

        std::vector<LandmarkObs> landmark_observations;
        for (int k = 0; k < observations.size(); k++) {
        	LandmarkObs obs = observations[k];
            double t_x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            double t_y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            landmark_observations.push_back(LandmarkObs{obs.id, t_x, t_y});
        }

        dataAssociation(predictions, landmark_observations);

        double weight = 1;
        vector<int> associations = vector<int>(landmark_observations.size());
        vector<double> sense_x = vector<double>(landmark_observations.size());
        vector<double> sense_y = vector<double>(landmark_observations.size());
        for (int l = 0; l < landmark_observations.size(); l++){
        	LandmarkObs obs = landmark_observations[l];
            LandmarkObs pred = pred_map[obs.id];
            double p = exp(-pow(obs.x - pred.x, 2) / (2 * std_x_2) - pow(obs.y - pred.y, 2.0) /
                                                                       (2 * std_y_2)) /
                       std_x_y;
            weight *= p;
            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        particle.weight = weight;
        SetAssociations(particle, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// Generator
	default_random_engine gen;

    for (int i = 0; i < particles.size(); i++) {
        weights[i] = particles[i].weight;
    }

    discrete_distribution<> d(weights.begin(), weights.end());

    vector<Particle> new_particles;
    for (int i = 0; i < particles.size(); i++) {
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
