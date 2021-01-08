#include "function.hpp"


// grid mapping
void grid_mapping() {

	start = std::chrono::system_clock::now();

	// mapping
	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		// key computation
		dataset_pt[i].key_computation(cutoff * sampling_rate_cell);

		// create cell or update point-set
		auto itr = grid.find(dataset_pt[i].key);
		if (itr == grid.end()) {

			// not created
			cell c(i);
			grid.insert({ dataset_pt[i].key,c });
			grid_key_set.push_back(dataset_pt[i].key);
		}
		else {

			// update point-set
			itr->second.point_id_set.push_back(i);
		}
	}

	auto it = grid.begin();
	while (it != grid.end()) {

		it->second.pivot_idx_decision();

		// increment iterator
		++it;
	}

	end = std::chrono::system_clock::now();
	cpu_grid_mapping = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " grid mapping time: " << cpu_grid_mapping << "[microsec]\n\n";
}

// local-density computation
void local_density_computation() {

	start = std::chrono::system_clock::now();

	// iterate range search
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < dataset_pt.size(); ++i) {

			if (dataset_pt[i].flag_sample == 1) {

				// random value decision
				std::mt19937 mt(dataset_pt[i].id);
				std::uniform_real_distribution<> rnd(0, 0.9999);

				// local density computation
				std::vector<int> result;
				kdtree.radiusSearch(point_set[i], cutoff, dataset_pt[i].result, dataset_pt[i].distance_to_nn);
				dataset_pt[i].local_density += dataset_pt[i].result.size();
				dataset_pt[i].local_density += rnd(mt);

				if (dataset_pt[i].distance_to_nn < cutoff) dataset_pt[i].distance_to_nn = cutoff;

				// neighbor cell computation
				grid[dataset_pt[i].key].neighbor_cell_computation(dataset_pt[i].result, i);
			}
		}
	}

	end = std::chrono::system_clock::now();
	double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " local density computation time: " << t << "[microsec]\n\n";
	cpu_local_density += t;
}

// dependent point compputation
void dependent_point_computation() {

	start = std::chrono::system_clock::now();

	// determine dependent point (constant-order)
	dependent_point_computation_constant_order();

	// set temporal label
	label_propagation();

	dependent_point_computation_second_phase();

	end = std::chrono::system_clock::now();
	cpu_dependency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " dependent point computation time: " << cpu_dependency << "[microsec]\n\n";

	// sort by descending order of dependent distance
	std::sort(cluster_center_candidate.begin(), cluster_center_candidate.end(), desc_dependent_dist);

	for (unsigned int i = 0; i < cluster_center_candidate.size(); ++i) {

		if (cluster_center_candidate[i]->dependent_dist < delta_min) break;

		// label is itself
		cluster_center_candidate[i]->label = cluster_center_candidate[i]->id;
		cluster_centers.push_back(cluster_center_candidate[i]->id);
	}
}

// label propagation
void computation_label_propagation() {

	start = std::chrono::system_clock::now();

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < cluster_centers.size(); ++i) {

			// init stack
			std::deque<unsigned int> stack;
			stack.push_back(cluster_centers[i]);

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
		}
	}

	end = std::chrono::system_clock::now();
	cpu_label = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " label propagation time: " << cpu_label << "[microsec]\n\n";

//	output_coord_label();
}


int main() {

	// parameter input
	input_parameter();

	// data input
	input_data();

	// inout label (ground trueth)
	input_label();

	// build kd-tree for range search
	kdtree_build();

	// display parameters
	std::cout << "-----------------\n";
	std::cout << " data id: " << dataset_id << "\n"
		<< " dimensionality: " << dimensionality << "\n"
		<< " sampling rate: " << sampling_rate << "\n"
		<< " sampling rate (cell): " << sampling_rate_cell << "\n"
		<< " cutoff-distance: " << cutoff << "\n"
		<< " core number: " << core_no << "\n";
	std::cout << "-----------------\n\n";

	// grid mapping
	grid_mapping();

	// local density compuation for small cells
	local_density_computation();

	// dependent point computation
	dependent_point_computation();

	// cluster assignment
	computation_label_propagation();

	// compute rand index
	compute_rand_index();

	// output computation time
	output_cpu_time();

//	output_decision_graph();
//	output_coord_label();

	return 0;
}
