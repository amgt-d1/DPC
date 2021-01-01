#include "file_io.hpp"


// unsigned int nonnoise_cnt = 0;


// distance computation
float computation_distance(pt& p, pt& q) {

	float dist = 0;
	for (unsigned int i = 0; i < dimensionality; ++i) dist += (p[i] - q[i]) * (p[i] - q[i]);
	return sqrt(dist);
}

// kd-tree init
void kdtree_build() {

	start = std::chrono::system_clock::now();
	kdtree.build(dataset_pt);
	end = std::chrono::system_clock::now();
	cpu_offline = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// local density check
void computation_local_density() {

	start = std::chrono::system_clock::now();

	// iterate range search
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < dataset_pt.size(); ++i) {

			std::mt19937 mt(dataset_pt[i].id);
			std::uniform_real_distribution<> rnd(0, 0.9999);

			dataset_pt[i].local_density = (kdtree.radiusSearch(dataset_pt[i], cutoff)).size();
			dataset_pt[i].local_density += rnd(mt);
		}
	}

	end = std::chrono::system_clock::now();
	double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " local density computation time: " << t << "[microsec]\n\n";

	cpu_local_density = t;

	// (parallel) sort by local density
	start = std::chrono::system_clock::now();

	std:sort(dataset_pt.begin(), dataset_pt.end(), desc_local_density);

	end = std::chrono::system_clock::now();
	t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	cpu_local_density += t;
}

// dependency check
void computation_dependency() {

	std::deque<double> array_time;

	start = std::chrono::system_clock::now();

	std::vector<unsigned int> buf;

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		//++nonnoise_cnt;

		if (dataset_pt[i].local_density < local_density_min) break;

		// NN search
		if (i >= 1) {
			spatial::neighbor_iterator<container_type> iter = neighbor_begin(kd_tree, dataset_pt[i]);
			dataset_pt[i].NN_dist = distance(iter);

			// store reverse NN
			rnn[iter->id].push_back(i);
		}
		else {
			dataset_pt[i].NN_dist = 0;
			float temp = 0;
			for (unsigned int j = 1; j < dataset_pt.size(); ++j) {
				temp = computation_distance(dataset_pt[i], dataset_pt[j]);
				if (temp > dataset_pt[i].NN_dist) dataset_pt[i].NN_dist = temp;
			}
		}

		// check cluster center
		if (delta_min <= dataset_pt[i].NN_dist && dataset_pt[i].local_density >= local_density_min) {

			// set label as itsself
			dataset_pt[i].label = dataset_pt[i].id;

			// store cluster center index
			cluster_centers.push_back(i);
		}

		// insert
		if (i < dataset_pt.size() - 1) {
			if (dataset_pt[i + 1].local_density == dataset_pt[i].local_density) {
				buf.push_back(i);
			}
			else {
				for (unsigned int j = 0; j < buf.size(); ++j) kd_tree.insert(dataset_pt[buf[j]]);
				buf.clear();

				kd_tree.insert(dataset_pt[i]);
			}
		}
	}
	end = std::chrono::system_clock::now();
	double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " dependency computation time: " << t << "[microsec]\n\n";

	//unsigned int noise_cnt = dataset_pt.size() - nonnoise_cnt;
	//std::cout << " noise count: " << noise_cnt << "\t" << "noise ratio: " << (double)noise_cnt / dataset_pt.size() << "\n\n";

	cpu_dependency = t;

	// output decision graph
//	output_decision_graph();
}

// label propagation
void computation_label_propagation() {

	start = std::chrono::system_clock::now();

	// init stack
	std::deque<unsigned int> stack;
	for (unsigned int i = 0; i < cluster_centers.size(); ++i) stack.push_back(cluster_centers[i]);

	unsigned int idx = 0;
	unsigned int id = 0;
	unsigned int label = 0;

	// depth-first traversal
	while (stack.size() > 0) {

		// get index of top element
		idx = stack[0];

		// get id of top element
		id = dataset_pt[idx].id;

		// set label
		label = dataset_pt[idx].label;

		// delete top element
		stack.pop_front();

		if (rnn.find(id) != rnn.end()) {

			for (unsigned int i = 0; i < rnn[id].size(); ++i) {

				if (dataset_pt[rnn[id][i]].label == -1) {

					// propagate label
					dataset_pt[rnn[id][i]].label = label;

					// update stack
					stack.push_front(rnn[id][i]);
				}
			}
		}
	}

	end = std::chrono::system_clock::now();
	cpu_label = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " label propagation time: " << cpu_label << "[microsec]\n\n";
}


int main() {

	// parameter input
	input_parameter();

	// data input
	input_data();

	// build kd-tree for range search
	kdtree_build();

	std::cout << " ---------\n";
	std::cout << " data id: " << dataset_id << "\n";
	std::cout << " dimensionality: " << dimensionality << "\n";
	std::cout << " sampling rate: " << sampling_rate << "\n";
	std::cout << " cutoff-disntance: " << cutoff << "\n";
	std::cout << " #threads: " << core_no << "\n";
	std::cout << " ---------\n\n";


	// local density computation
	computation_local_density();

	// dependency check & cluster center identification
	computation_dependency();

	// cluster assignment
	computation_label_propagation();

	// output computation time
	output_cpu_time();

	// output label
//	output_label();
//	output_coord_label();

	return 0;
}