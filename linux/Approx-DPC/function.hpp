#include "file_io.hpp"
#include <deque>
#include <omp.h>


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

// kd-tree build (spatial lib.)
void kd_tree_build(container_type& kd_tree, std::vector<pts>& pts) {

	for (unsigned int i = 0; i < pts.size(); ++i) kd_tree.insert(pts[i]);
}

// cost-based grid partition function
void grid_partition_by_cost(std::vector<sc_pair>& grid, std::vector<std::vector<std::string>>& grid_partition) {

	// clear g_p
	grid_partition.clear();

	// sort grid by non-increasing order (to get 3/2-approximation)
	std::sort(grid.begin(), grid.end(), std::greater<sc_pair>());

	// cost array (each element suggests the temporal cost of each thread)
	std::vector<double> cost_array(core_no);

	// init min-cost & thread id
	double cost_min = 10000000000;
	unsigned int thread_id = 0;

	// resize
	grid_partition.resize(core_no);

	// partition grid
	for (unsigned int i = 0; i < grid.size(); ++i) {

		// compute the thread with the minimum cost
		for (unsigned int j = 0; j < core_no; ++j) {

			if (cost_array[j] < cost_min) {
				thread_id = j;
				cost_min = cost_array[j];
			}
		}

		// assign cell
		grid_partition[thread_id].push_back(grid[i].key);

		// update cost array
		cost_array[thread_id] += grid[i].cost;

		// update cost_min (upper-bound)
		cost_min = cost_array[thread_id];
	}
}

// constant-order approach for dependent point computation
void dependent_point_computation_constant_order() {

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < dataset_pt.size(); ++i) {

			if (dataset_pt[i].local_density > local_density_min) {

				// access its own cell
				auto itr = grid.find(dataset_pt[i].key);

				// point with non-max local density in the cell
				if (dataset_pt[i].id != itr->second.id_max) {

					// dependent-dist <- cutoff
					dataset_pt[i].dependent_dist = cutoff;

					// determine dependent point
					dataset_pt[i].dependent_point_id = itr->second.id_max;
				}
				else {

					// point with max local density in the cell
					auto itr_neighbor_index = itr->second.neighbor_index.begin();
					while (itr_neighbor_index != itr->second.neighbor_index.end()) {

						if (dataset_pt[i].local_density < grid[*itr_neighbor_index].min_local_density) {

							// dependent-dist <- cutoff
							dataset_pt[i].dependent_dist = cutoff;

							// determine dependent point
							dataset_pt[i].dependent_point_id = grid[*itr_neighbor_index].id_max;

							break;
						}

						++itr_neighbor_index;
					}
				}
			}
		}
	}
}

// kd-forest build (spatial)
void kdforest_build(std::vector<unsigned int>& idx, std::vector<container_type> &kdforest, std::vector<std::vector<pts>>& dataset_subsets) {

	// compute subset size
	//float subsets = powf(dataset_pt.size(), (1.0 / dimensionality));
	//subsets = powf(subsets, (float)dimensionality / (dimensionality + 1));
	//subset_size = (unsigned int)subsets + 1;
	subset_size = std::log2f((float)dataset_pt.size());
	//std::cout << " s: " << subset_size << "\n";

	// variables for partition
	std::vector<pts> subset;
	unsigned int subset_cardinliaty = dataset_pt.size() / subset_size;
	unsigned int subset_counter = 0;
	if (dataset_pt.size() % subset_size > 0) ++subset_cardinliaty;

	// reduction & equal partition
	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		if (dataset_pt[i].local_density > local_density_min) {

			// subset update
			subset.push_back(point_set[i]);
			if (subset.size() == subset_cardinliaty) {
				dataset_subsets.push_back(subset);
				subset.clear();
			}

			// reverse NN update & idx insertion
			if (dataset_pt[i].dependent_dist == cutoff) {

				// reverse nn update
				rnn[dataset_pt[i].dependent_point_id].push_back(dataset_pt[i].id);
			}
			else {

				// idx insertion
				idx.push_back(i);
			}
		}
		else {

			// subset update
			dataset_subsets.push_back(subset);

			break;
		}
	}

	// build kd-tree for each subset
	kdforest.resize(dataset_subsets.size());

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < dataset_subsets.size(); ++i) {
			kd_tree_build(kdforest[i], dataset_subsets[i]);
		}
	}
}

// partition set of points whose dependent points could not determined
void idx_partition(std::vector<unsigned int>& idx, std::vector<std::vector<pts>>& dataset_subsets, std::vector<std::vector<unsigned int>>& idx_partitioned) {

	// cost calculation
	std::vector<std::pair<float, unsigned int>> idx_cost;
	idx_cost.resize(idx.size());

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < idx.size(); ++i) {

			idx_cost[i].second = idx[i];
			unsigned int count_tree = 0;
			unsigned int count_scan = 0;

			for (int j = 0; j < dataset_subsets.size(); ++j) {

				if (dataset_subsets[j][0].local_density <= dataset_pt[idx[i]].local_density) {
					break;
				}
				else {

					if (dataset_subsets[j][dataset_subsets[j].size() - 1].local_density > dataset_pt[idx[i]].local_density) {
						++count_tree;
					}
					else {
						++count_scan;
					}
				}
			}

			// cost calculation
			idx_cost[i].first = (count_scan * dataset_subsets[0].size()) + (count_tree * powf(dataset_subsets[0].size(), 1.0 - 1.0 / dimensionality));
		}
	}

	// sort
	std::sort(idx_cost.begin(), idx_cost.end(), std::greater<std::pair<float, unsigned int>>());

	// greedy-partition
	idx_partitioned.resize(core_no);
	std::vector<float> array_cost;
	array_cost.resize(core_no);

	for (unsigned int i = 0; i < idx_cost.size(); ++i) {

		
		float cost_min = FLT_MAX;
		unsigned int id_cost_min = 0;
		for (unsigned int j = 0; j < core_no; ++j) {
			if (array_cost[j] < cost_min) {
				cost_min = array_cost[j];
				id_cost_min = j;
			}
		}

		// update array_cost
		array_cost[id_cost_min] += idx_cost[i].first;

		// assign idx
		idx_partitioned[id_cost_min].push_back(idx_cost[i].second);
	}	
}

// index-based dependent point computation (spatial-theoretical)
void dependent_point_computation_index(std::vector<std::vector<unsigned int>>& idx_partitioned, std::vector<std::vector<pts>>& dataset_subsets, std::vector<container_type> &kdforest) {

	unsigned int count = 0;
	unsigned int count_ = 0;

	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < idx_partitioned.size(); ++i) {

			for (unsigned int j = 0; j < idx_partitioned[i].size(); ++j) {

				for (unsigned int k = 0; k < dataset_subsets.size(); ++k) {

					if (dataset_subsets[k][0].local_density <= dataset_pt[idx_partitioned[i][j]].local_density) {
						break;
					}
					else {

						count_ = count_ + 1;

						if (dataset_subsets[k][dataset_subsets[k].size() - 1].local_density > point_set[idx_partitioned[i][j]].local_density) {

							// NN-search on kd-tree
							spatial::neighbor_iterator<container_type> iter = neighbor_begin(kdforest[k], point_set[idx_partitioned[i][j]]);
							float dist = distance(iter);
							if (dist < dataset_pt[idx_partitioned[i][j]].dependent_dist) {
								dataset_pt[idx_partitioned[i][j]].dependent_dist = dist;
								dataset_pt[idx_partitioned[i][j]].dependent_point_id = iter->id;
							}
						}
						else {

							// NN-search by scan
							for (unsigned int l = 0; l < dataset_subsets[k].size(); ++l) {

								if (dataset_subsets[k][l].local_density > point_set[idx_partitioned[i][j]].local_density) {
									float dist = computation_distance(dataset_subsets[k][l], point_set[idx_partitioned[i][j]]);
									if (dist < dataset_pt[idx_partitioned[i][j]].dependent_dist) {
										dataset_pt[idx_partitioned[i][j]].dependent_dist = dist;
										dataset_pt[idx_partitioned[i][j]].dependent_point_id = dataset_subsets[k][l].id;
									}
								}
								else {
									break;
								}
							}
						}
					}
				}

				if (dataset_pt[idx_partitioned[i][j]].dependent_dist <= cutoff) count = count + 1;

				// check cluster center
				if (dataset_pt[idx_partitioned[i][j]].dependent_dist >= delta_min) {

					// set label as itsself
					dataset_pt[idx_partitioned[i][j]].label = dataset_pt[idx_partitioned[i][j]].id;

					// store cluster center
					#pragma omp critical
					cluster_centers.push_back(dataset_pt[idx_partitioned[i][j]].id);
				}
			}
		}
	}
}
