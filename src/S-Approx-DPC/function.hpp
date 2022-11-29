#include "file_io.hpp"
#include <functional>
#include <deque>
#include <omp.h>
#include <cfloat>


std::vector<unsigned int> cluster_centers_temporal;
std::vector<std::vector<pt*>> cluster_temporal;
std::unordered_map<unsigned int, unsigned int> label_temporal_table;	// key is label, value is idx of cluster_temporal
std::unordered_map<unsigned int, float> cluster_radius;
std::vector<pt*> cluster_center_candidate;

// kd-tree init
void kdtree_build() {

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {
		pts p(dataset_pt[i].id, dataset_pt[i]);
		point_set.push_back(p);
	}

	start = std::chrono::system_clock::now();
	kdtree.build(point_set);
	end = std::chrono::system_clock::now();
	cpu_offline = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// constant-order approach for dependet point computation
void dependent_point_computation_constant_order() {

	#pragma omp parallel num_threads(core_no)
	{
		// get dependent point within d_cut
		#pragma omp for schedule(static)
		for (int i = 0; i < dataset_pt.size(); ++i) {

			// get cell
			auto itr = grid.find(dataset_pt[i].key);

			if (dataset_pt[i].flag_sample) {

				if (dataset_pt[i].local_density > local_density_min) {

					bool flag = 1;

					// iterate neighbor cell
					auto itr_neighbor = itr->second.neighbor_cell.begin();
					while (itr_neighbor != itr->second.neighbor_cell.end()) {

						// get sampled point
						const unsigned int idx = grid[*itr_neighbor].point_id_set[0];

						// check local density
						if (dataset_pt[i].local_density < dataset_pt[idx].local_density) {

							// distance computation
							const float distance = computation_distance(dataset_pt[i], dataset_pt[idx]);

							if (distance < cutoff) {

								flag = 0;

								// update dependent point
								if (dataset_pt[i].dependent_dist > distance) {

									dataset_pt[i].dependent_dist = distance;
									dataset_pt[i].dependent_point_id = idx;
								}

								// check termination condition
								if (dataset_pt[i].dependent_dist <= (1.0 + sampling_rate_cell) * dataset_pt[i].distance_to_nn) break;
							}
						}
						else {
							
							// update flag
							flag = 0;
						}

						if (flag) {
							++itr_neighbor;
						}
						else {
							itr->second.neighbor_cell.erase(itr_neighbor++);
						}
					}

					if (dataset_pt[i].dependent_dist == dist_max) {

						if (itr->second.neighbor_cell.size() > 0) {

							itr_neighbor = itr->second.neighbor_cell.begin();

							// get sampled point
							const unsigned int idx = grid[*itr_neighbor].point_id_set[0];

							// distance computation
							const float distance = computation_distance(dataset_pt[i], dataset_pt[idx]);

							// update dependent point
							if (dataset_pt[i].dependent_dist > distance) {

								dataset_pt[i].dependent_dist = distance;
								dataset_pt[i].dependent_point_id = idx;
							}
						}
						else {

							// set temporal label
							dataset_pt[i].label = dataset_pt[i].id;

							#pragma omp critical
							cluster_centers_temporal.push_back(dataset_pt[i].id);
						}
					}
				}
			}
			else {

				unsigned int idx = itr->second.pivot_idx;

				if (dataset_pt[itr->second.point_id_set[itr->second.pivot_idx]].local_density > local_density_min) {

					// dependent-dist <- cutoff * epsilon
					dataset_pt[i].dependent_dist = cutoff * sampling_rate_cell;

					// determine dependent point
					dataset_pt[i].dependent_point_id = itr->second.point_id_set[itr->second.pivot_idx];
				}
			}
		}
	}

	// temporal rnn comptation
	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		if (dataset_pt[i].dependent_dist < dist_max) rnn[dataset_pt[i].dependent_point_id].push_back(i);
	}
}

// temporal label propagation
void label_propagation() {

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < cluster_centers_temporal.size(); ++i) {

			// init stack
			std::deque<unsigned int> stack;
			stack.push_back(cluster_centers_temporal[i]);

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

	// summarize temporal clusters
	cluster_temporal.resize(cluster_centers_temporal.size());
	for (unsigned int i = 0; i < cluster_centers_temporal.size(); ++i) label_temporal_table[cluster_centers_temporal[i]] = i;

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		if (dataset_pt[i].label != -1) cluster_temporal[label_temporal_table[dataset_pt[i].label]].push_back(&dataset_pt[i]);
	}

	// get cluster radius
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < cluster_temporal.size(); ++i) {

			// sort
			std::sort(cluster_temporal[i].begin(), cluster_temporal[i].end(), desc_local_density_);

			float radius_max = 0;
			float distance = 0;
			int label = cluster_temporal[i][0]->label;

			for (unsigned int j = 0; j < cluster_temporal[i].size(); ++j) {

				if (cluster_temporal[i][j]->local_density > 0) {
					distance = computation_distance(dataset_pt[label], *cluster_temporal[i][j]);
					if (distance > radius_max) radius_max = distance;
				}

				// init label
				cluster_temporal[i][j]->label = -1;
			}

			#pragma omp critical
			cluster_radius[label] = radius_max;
		}
	}
}

// dependent point computation (the second phase)
void dependent_point_computation_second_phase() {

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < cluster_centers_temporal.size(); ++i) {

			// get idx of temporal cluster center
			unsigned int idx_c = cluster_centers_temporal[i];

			// get temporal cluster center with higher density
			float distance_min = FLT_MAX;
			std::unordered_map<unsigned int, float> distance_to_center;	// key: idx (not id), map: distance
			float distance = 0;

			for (unsigned int j = 0; j < cluster_centers_temporal.size(); ++j) {

				if (dataset_pt[idx_c].local_density < dataset_pt[cluster_centers_temporal[j]].local_density) {

					distance = computation_distance(dataset_pt[idx_c], dataset_pt[cluster_centers_temporal[j]]);
					distance_to_center[j] = distance;
					if (distance_min > distance) distance_min = distance;
				}
			}

			// shrink candidate size by triangle inequality
			std::vector<unsigned int> candidate;
			auto it = distance_to_center.begin();
			while (it != distance_to_center.end()) {

				if (it->second - cluster_radius[cluster_centers_temporal[it->first]] <= distance_min) candidate.push_back(it->first);

				++it;
			}

			// compute denpendent dist
			for (unsigned int j = 0; j < candidate.size(); ++j) {

				// get idx of compared cluster
				unsigned int idx = label_temporal_table[cluster_centers_temporal[candidate[j]]];

				for (unsigned int k = 0; k < cluster_temporal[idx].size(); ++k) {

					if (dataset_pt[idx_c].local_density < cluster_temporal[idx][k]->local_density) {

						distance = computation_distance(dataset_pt[idx_c], *cluster_temporal[idx][k]);

						// update dependent distance and point
						if (distance < dataset_pt[idx_c].dependent_dist) {
							dataset_pt[idx_c].dependent_dist = distance;
							dataset_pt[idx_c].dependent_point_id = cluster_temporal[idx][k]->id;
						}
					}
					else {
						break;
					}
				}
			}

			// rnn update
			if (dataset_pt[idx_c].dependent_dist < dist_max) {

				#pragma omp critical
				rnn[dataset_pt[idx_c].dependent_point_id].push_back(idx_c);
			}

			// insert into cluster center candidate
			#pragma omp critical
			cluster_center_candidate.push_back(&dataset_pt[idx_c]);
		}
	}
}