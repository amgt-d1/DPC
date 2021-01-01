#include <random>
#include <array>
#include <unordered_set>
#include <unordered_map>
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

	// sample flag
	bool flag_sample;

	float distance_to_nn = dist_max;

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
		flag_sample = 0;
	}

	/* constructor */
	pt(unsigned int id_, std::array<float, dimensionality>& ar) {
		id = id_;
		dependent_dist = dist_max;
		dependent_point_id = 0;
		local_density = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
		label = -1;
		key = "";
		flag_sample = 0;
	}

	/*******************/
	/* member function */
	/*******************/

	// identifier update
	void update_id(unsigned int id_) { this->id = id_; }

	// coordinate update
	void update_val(std::array<float, dimensionality>& ar) {
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
		result.shrink_to_fit();
	}
};

// sort descending order of identifier
bool asc_id(const pt& l, const pt& r) {
	return l.id < r.id;
}

// sort descending order of local density
bool desc_local_density(const pt& l, const pt& r) {
	return l.local_density > r.local_density;
}

// sort descending order of local density
bool desc_local_density_(const pt* l, const pt* r) {
	return l->local_density > r->local_density;
}

// sort descending order of depedent dist
bool desc_dependent_dist(const pt* l, const pt* r) {
	return l->dependent_dist > r->dependent_dist;
}

// distance computation for pt
float computation_distance(pt& p, pt& q) {

	float dist = 0;
	for (unsigned int i = 0; i < dimensionality; ++i) dist += (p[i] - q[i]) * (p[i] - q[i]);
	return sqrt(dist);
}


// definition of "cell"
class cell {
public:

	// a set of mapped point identifiers
	std::vector<unsigned int> point_id_set;

	// neighbor cell key
	std::unordered_set<std::string> neighbor_cell;

	unsigned int pivot_idx = 0;


	/* constructor */
	cell() {}

	/* constructor */
	cell(unsigned int id) { point_id_set.push_back(id);	}

	/*******************/
	/* member function */
	/*******************/

	// compute neighbor cells
	void neighbor_cell_computation(std::vector<int> &result, unsigned int id) {

		for (unsigned int i = 0; i < result.size(); ++i) neighbor_cell.insert(dataset_pt[result[i]].key);
		neighbor_cell.erase(dataset_pt[id].key);

		// release memory
		dataset_pt[id].result_release();
	}

	// decide pivot idx
	void pivot_idx_decision() {

		// random value decision
		std::mt19937 mt(12);
		std::uniform_int_distribution<> rnd(0, point_id_set.size() - 1);

		dataset_pt[point_id_set[pivot_idx]].flag_sample = 1;
	}
};

// grid
std::unordered_map<std::string, cell> grid;
std::vector<std::string> grid_key_set;

