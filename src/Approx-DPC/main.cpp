#include "function.hpp"


// data structure definition
std::vector<sc_pair> grid_m, grid_l;
std::vector<std::vector<std::string>> grid_many, grid_less;


// grid mapping
void grid_mapping() {

	start = std::chrono::system_clock::now();

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {

		// key computation
		dataset_pt[i].key_computation(cutoff);

		// create cell or update point-set
		auto itr = grid.find(dataset_pt[i].key);
		if (itr == grid.end()) {

			// not created
			cell c(i);
			grid.insert({ dataset_pt[i].key,c });
		}
		else {

			// update point-set
			itr->second.point_id_set.push_back(i);
		}
	}

	end = std::chrono::system_clock::now();
	cpu_grid_mapping = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " grid mapping time: " << cpu_grid_mapping << "[microsec]\t" << "grid size: " << grid.size() << "\n\n";
}

// grid partition (small and large cells)
void grid_partition_small_and_large() {

	start = std::chrono::system_clock::now();

	// partition into small and large
	auto itr = grid.begin();
	while (itr != grid.end()) {

		if (itr->second.point_id_set.size() < cell_pts_size_min) {

			// small case
			grid_l.push_back({ itr->first, (double)(itr->second.point_id_set.size() + 1) });
			for (unsigned int i = 0; i < itr->second.point_id_set.size(); ++i) dataset_pt[itr->second.point_id_set[i]].flag = 1;
		}
		else {

			// large case
			grid_m.push_back({ itr->first, (double)itr->second.point_id_set.size() });
		}

		++itr;
	}

	end = std::chrono::system_clock::now();
	cpu_grid_partition = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " grid partition time: " << cpu_grid_partition << "[microsec]\n\n";
}

// local-density computation for small cells
void local_density_computation_small() {

	start = std::chrono::system_clock::now();

	// iterate range search
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < dataset_pt.size(); ++i) {

			if (dataset_pt[i].flag == 1) {

				// random value decision
				std::mt19937 mt(dataset_pt[i].id);
				std::uniform_real_distribution<> rnd(0, 0.9999);

				// local density computation
				dataset_pt[i].result = kdtree.radiusSearch(point_set[i], cutoff);
				dataset_pt[i].local_density = dataset_pt[i].result.size();
				dataset_pt[i].local_density += rnd(mt);

				point_set[i].local_density = dataset_pt[i].local_density;
			}
		}

		// neighbor-index building
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < grid_l.size(); ++i) {
			grid[grid_l[i].key].neighbr_index_build();
		}
	}

	end = std::chrono::system_clock::now();
	double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	cpu_local_density += t;
}

// local-density computation for large cells
void local_density_computation_large() {

	start = std::chrono::system_clock::now();

	// partition large grid
	grid_partition_by_cost(grid_m, grid_many);

	// candidate computation & cost update
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < core_no; ++i) {

			// candidate computation
			for (unsigned int j = 0; j < grid_many[i].size(); ++j) grid[grid_many[i][j]].candidate_computation(cutoff);
		}

		#pragma omp for schedule(static)
		for (int i = 0; i < grid_m.size(); ++i) {

			// cost update
			grid_m[i].cost = (double)((grid[grid_m[i].key].candaidate_id_set.size() + 1) * grid[grid_m[i].key].point_id_set.size());
		}
	}

	// re-partition
	grid_partition_by_cost(grid_m, grid_many);

	// local-density computation
	#pragma omp parallel num_threads(core_no)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < core_no; ++i) {

			// local-density comp. by scan
			for (unsigned int j = 0; j < grid_many[i].size(); ++j) grid[grid_many[i][j]].joint_local_density_computation(cutoff);
		}
	}

	end = std::chrono::system_clock::now();
	double t = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	cpu_local_density += t;
	std::cout << " local density computation time (2): " << cpu_local_density << "[microsec]\n\n";
}

// dependent point computation
void dependent_point_computation() {

	start = std::chrono::system_clock::now();

	// determine dependent point (constant-order)
	dependent_point_computation_constant_order();

	// sort by local density
	std::sort(dataset_pt.begin(), dataset_pt.end(), desc_local_density);
	std::sort(point_set.begin(), point_set.end(), desc_local_density_pts);

	// building kd-tree for each subset
	std::vector<unsigned int> idx;					// a set of points which could not determine their dependent point
	std::vector<std::vector<pts>> dataset_subsets;	// partition result
	std::vector<container_type> kdforest;
	kdforest_build(idx, kdforest, dataset_subsets);

	// cost-based partition
	std::vector<std::vector<unsigned int>> idx_partitioned;
	idx_partition(idx, dataset_subsets, idx_partitioned);

	// determine dependent point (index-based)
	dependent_point_computation_index(idx_partitioned, dataset_subsets, kdforest);

	// reduction
	for (unsigned int i = 0; i < idx.size(); ++i) {
		if (dataset_pt[idx[i]].dependent_dist != dist_max && dataset_pt[idx[i]].local_density >= local_density_min) rnn[dataset_pt[idx[i]].dependent_point_id].push_back(dataset_pt[idx[i]].id);
	}

	end = std::chrono::system_clock::now();
	cpu_dependency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << " dependent point computation time: " << cpu_dependency << "[microsec]\n\n";
}

// label propagation
void computation_label_propagation() {

	start = std::chrono::system_clock::now();

	// sort by ID
	std::sort(dataset_pt.begin(), dataset_pt.end(), asc_id);

	// init stack
	std::deque<unsigned int> stack;
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < cluster_centers.size(); ++i) {
		stack.push_back(cluster_centers[i]);
		dataset_pt[cluster_centers[i]].label = cnt;
		++cnt;
	}

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

	// exact label input
	//input_label();

	// build kd-tree for range search
	kdtree_build();

	// display current time & parameters
	std::cout << " -----------\n";
	std::cout << " data id: " << dataset_id << "\n";
	std::cout << " dimensionality: " << dimensionality << "\n";
	std::cout << " sampling rate: " << sampling_rate << "\n";
	std::cout << " cardinality: " << dataset_pt.size() << "\n";
	std::cout << " cutoff-distance: " << cutoff << "\n";
	std::cout << " thread number: " << core_no << "\n";
	std::cout << " -----------\n\n";

	// grid mapping
	grid_mapping();


	// grid partition
	grid_partition_small_and_large();

	// local density computation for small cells
	local_density_computation_small();

	// local density computation for large cells
	local_density_computation_large();

	// dependent point computation
	dependent_point_computation();

	// cluster assignment
	computation_label_propagation();

	// output computation time
	output_cpu_time();

	//output_label_();

	return 0;
}
