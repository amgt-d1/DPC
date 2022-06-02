#include "../spatial_2.1.8/src/point_multiset.hpp"	// change the path based on your environment
#include "../spatial_2.1.8/src/idle_point_multiset.hpp"
#include "../spatial_2.1.8/src/bits/spatial_neighbor.hpp"
#include <array>
#include <unordered_map>
#include "kdtree.hpp"


// parameter
const unsigned int dimensionality = 2;


// definition
class pt : public std::array<float, dimensionality> {

public:
	static const int DIM = dimensionality;

	// point id
	unsigned int id;

	// local density (#points within a range)
	float local_density;

	// distance to NN with higher local density
	float NN_dist;

	// cluster label
	int label;


	/* constructor */
	pt() {
		id = 0;
		NN_dist = 1000000000;
		local_density = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = 0;
		label = -1;
	}

	/* constructor */
	pt(unsigned int id_, std::array<float, dimensionality> & ar) {
		id = id_;
		NN_dist = 1000000000;
		local_density = 0;
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
		label = -1;
	}

	/* member function */
	void update_id(unsigned int id_) { this->id = id_; }

	void update_val(std::array<float, dimensionality> & ar) {
		for (unsigned int i = 0; i < dimensionality; ++i) (*this)[i] = ar[i];
	}

};

typedef std::array<float, dimensionality> point;
typedef spatial::idle_point_multiset<dimensionality, pt> container_type;

// kd-tree
kdt::KDTree<pt> kdtree;	// for range search
container_type kd_tree;	// for NN search and insertion

// dataset
std::vector<pt> dataset_pt;

// reverse nn map
std::unordered_map<unsigned int, std::vector<unsigned int>> rnn;	// key: id, value: reverse NNs


// sort descending order of local density
bool desc_local_density(const pt & l, const pt & r) {
	return l.local_density > r.local_density;
}

// sort descending order of dependent distance
bool desc_dependency_distance(const pt & l, const pt & r) {
	return l.NN_dist > r.NN_dist;
}
