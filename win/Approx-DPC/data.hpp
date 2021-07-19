#include <src/idle_point_multiset.hpp>
#include <src\bits\spatial_neighbor.hpp>
#include <random>
#include <array>
#include <unordered_set>
#include <unordered_map>
//#include <concurrent_vector.h>
//#include <concurrent_unordered_map.h>
#include "kdtree.hpp"


// parameter & variable
const unsigned int dimensionality = 2;
const float dist_max = 1000000000;

// decreare class "pt"
class pt;

// decreare class "pts" which consists only of vector
class pts;

// kd-tree
kdt::KDTree<pts> kdtree;

// dataset
std::vector<pt> dataset_pt;
std::vector<pts> point_set;

// reverse NN map
std::unordered_map<unsigned int, std::vector<unsigned int>> rnn;	// key: id, value: reverse NNs


// definition of "point"
typedef std::array<float, dimensionality> point;

// definition of "pts"
class pts : public std::array<float, dimensionality> {

public:
	static const int DIM = dimensionality;

	// point id
	unsigned int id;

	// local density
	float local_density = 0;


	/* constructor */
	pts() {
		id = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = 0;
	}

	/* constructor */
	pts(unsigned int id_, std::array<float, dimensionality>& ar) {
		id = id_;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
	}

	pts(unsigned int id_, std::array<float, dimensionality>& ar, float rho) {
		id = id_;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
		local_density = rho;
	}

};


// definition of "pt"
class pt : public std::array<float, dimensionality> {

public:
	static const int DIM = dimensionality;

	// point id
	unsigned int id;

	// local density (#points within a range)
	float local_density;

	// distance to NN with higher local density
	float dependent_dist;

	// id of dependent point
	unsigned int dependent_point_id;

	// cluster label
	int label;

	// cell key
	std::string key;

	// flag for local densoty < cell_pts_min_size
	bool flag;

	// result of range search
	std::vector<int> result;


	/* constructor */
	pt() {
		id = 0;
		dependent_dist = dist_max;
		dependent_point_id = 0;
		local_density = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = 0;
		label = -1;
		key = "";
		flag = 0;
	}

	/* constructor */
	pt(unsigned int id_, std::array<float, dimensionality> & ar) {
		id = id_;
		dependent_dist = dist_max;
		dependent_point_id = 0;
		local_density = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
		label = -1;
		key = "";
		flag = 1;
	}

	/*******************/
	/* member function */
	/*******************/

	// identifier update
	void update_id(unsigned int id_) { this->id = id_; }

	// coordinate update
	void update_val(std::array<float, dimensionality> & ar) {
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
	}

	// cell key computation
	void key_computation(float cutoff) {

		const float divide = cutoff / sqrt(dimensionality);
		unsigned int temp = 0;

		for (unsigned int i = 0; i < dimensionality; ++i) {
			temp = (*this)[i] / divide;
			key += std::to_string(temp) + "+";
		}
	}

	// memory release
	void result_release() {
		std::vector<int>().swap(result);
	}
};

// sort descending order of local density
bool asc_id(const pt& l, const pt& r) {
	return l.id < r.id;
}

// sort descending order of local density
bool desc_local_density(const pt & l, const pt & r) {
	return l.local_density > r.local_density;
}

// sort descending order of local density
bool desc_local_density_pts(const pts& l, const pts& r) {
	return l.local_density > r.local_density;
}

// distance computation for pt
float computation_distance(pt& p, pt& q) {

	float dist = 0;
	for (unsigned int i = 0; i < dimensionality; ++i) dist += (p[i] - q[i]) * (p[i] - q[i]);
	return sqrt(dist);
}

// distance computation for pts
float computation_distance(pts& p, pts& q) {

	float dist = 0;
	for (unsigned int i = 0; i < dimensionality; ++i) dist += (p[i] - q[i]) * (p[i] - q[i]);
	return sqrt(dist);
}


// definition of "cell"
class cell {
public:

	// a set of mapped point identifiers
	std::vector<unsigned int> point_id_set;

	// a candidate neighbor points
	std::vector<int> candaidate_id_set;

	// min local density
	float min_local_density;

	// identifier of point with max local-density
	unsigned int id_max;

	// neighbor-index
	std::unordered_set<std::string> neighbor_index;


	/* constructor */
	cell() {
		min_local_density = 10000000;
		id_max = 0;
	}

	/* constructor */
	cell(unsigned int id) {
		min_local_density = 10000000;
		id_max = 0;
		point_id_set.push_back(id);
	}

	/*******************/
	/* member function */
	/*******************/

	// neighbor-index build for cell with less points
	void neighbr_index_build() {

		float local_density_max = 0;

		for (unsigned int i = 0; i < point_id_set.size(); ++i) {

			// id_max update
			if (dataset_pt[point_id_set[i]].local_density > local_density_max) {

				// clear
				dataset_pt[id_max].result.clear();

				// update
				local_density_max = dataset_pt[point_id_set[i]].local_density;
				id_max = dataset_pt[point_id_set[i]].id;
			}

			// min rho update
			if (dataset_pt[point_id_set[i]].local_density < min_local_density) min_local_density = dataset_pt[point_id_set[i]].local_density;

			// clear
			if (dataset_pt[point_id_set[i]].id != id_max) dataset_pt[point_id_set[i]].result.clear();
		}

		// neighbor index build
		for (unsigned int i = 0; i < dataset_pt[id_max].result.size(); ++i) neighbor_index.insert(dataset_pt[dataset_pt[id_max].result[i]].key);
//		dataset_pt[id_max].neighbor_index_build(dataset_pt[id_max].result);
	}

	// make cell center point
	void make_center_point(pts &q, float cutoff) {

		const float divide = cutoff / sqrt(dimensionality);
		unsigned int temp = 0;

		for (unsigned int i = 0; i < dimensionality; ++i) {

			temp = dataset_pt[point_id_set[0]][i] / divide;
			q[i] = (float)(temp * divide);
			q[i] += divide / 2.0;
		}
	}

	// candidate computation
	void candidate_computation(float cutoff) {

		pts q;
		make_center_point(q, cutoff);

		float dist = 0;
		float dist_max = 0;

		for (unsigned int i = 0; i < point_id_set.size(); ++i) {
			dist = computation_distance(q, point_set[point_id_set[i]]);
			if (dist > dist_max) dist_max = dist;
		}

		candaidate_id_set = kdtree.radiusSearch(q, cutoff + dist_max);
	}

	// local-density computation
	void joint_local_density_computation(float cutoff) {

		float dist = 0;
		float local_density_max = 0;

		for (unsigned int i = 0; i < point_id_set.size(); ++i) {

			// random value decision
			std::mt19937 mt(dataset_pt[point_id_set[i]].id);
			std::uniform_real_distribution<> rnd(0, 0.9999);

			// local density computation
			for (unsigned int j = 0; j < candaidate_id_set.size(); ++j) {

				if (point_id_set[i] != candaidate_id_set[j]) {
					dist = computation_distance(point_set[point_id_set[i]], point_set[candaidate_id_set[j]]);
					if (dist < cutoff) ++dataset_pt[point_id_set[i]].local_density;
				}
			}
			dataset_pt[point_id_set[i]].local_density += rnd(mt);
			point_set[point_id_set[i]].local_density = dataset_pt[point_id_set[i]].local_density;

			// id_max update
			if (local_density_max < dataset_pt[point_id_set[i]].local_density) {

				// update
				local_density_max = dataset_pt[point_id_set[i]].local_density;
				id_max = dataset_pt[point_id_set[i]].id;
			}

			// min rho update
			if (dataset_pt[point_id_set[i]].local_density < min_local_density) min_local_density = dataset_pt[point_id_set[i]].local_density;
		}

		// neighbor-index building
		for (unsigned int i = 0; i < candaidate_id_set.size(); ++i) {
			if (id_max != candaidate_id_set[i]) {
				dist = computation_distance(point_set[id_max], point_set[candaidate_id_set[i]]);
				if (dist < cutoff) neighbor_index.insert(dataset_pt[candaidate_id_set[i]].key);
			}
		}

		// clear
		candaidate_id_set.clear();
	}
};


// grid
//concurrency::concurrent_unordered_map<std::string, cell> grid;
std::unordered_map<std::string, cell> grid;

// definition of "string-cost pair"
struct sc_pair {

	std::string key;
	double cost;

	bool operator>(const sc_pair& p) const {
		return cost > p.cost;
	}
};


// definition of kd-tree (spatial lib.)
typedef spatial::idle_point_multiset<dimensionality, pts> container_type;
container_type kd_tree;	// for NN search and insertion
