#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "data.hpp"
#include <chrono>
#include <random>
#include <deque>


// variable for time measure
std::chrono::system_clock::time_point start, end;

// dataset identifier
unsigned int dataset_id = 0;

// sampling rate
float sampling_rate = 1;

// cutoff distance
float cutoff = 0;
unsigned int local_density_min = 0;

// parameter for cluster center
float delta_min = 0;

// system parameter
unsigned int core_no = 1;

// result
double cpu_offline = 0;
double cpu_local_density = 0;
double cpu_dependency = 0;
double cpu_label = 0;
std::vector<unsigned int> cluster_centers;


// parameter input
void input_parameter() {

	std::ifstream ifs_cutoff("parameter/cutoff.txt");
	std::ifstream ifs_dataset_id("parameter/dataset_id.txt");
	std::ifstream ifs_core_no("parameter/thread_num.txt");
	std::ifstream ifs_sampling_rate("parameter/sampling_rate.txt");

	if (ifs_cutoff.fail()) {
		std::cout << " cutoff.txt does not exist." << std::endl;
		std::exit(0);
	}
	else if (ifs_dataset_id.fail()) {
		std::cout << " dataset_id.txt does not exist." << std::endl;
		std::exit(0);
	}
	else if (ifs_core_no.fail()) {
		std::cout << " thread_num.txt does not exist." << std::endl;
		std::exit(0);
	}
	else if (ifs_sampling_rate.fail()) {
		std::cout << " sampling_rate.txt does not exist." << std::endl;
		std::exit(0);
	}

	while (!ifs_cutoff.eof()) { ifs_cutoff >> cutoff; }
	while (!ifs_dataset_id.eof()) { ifs_dataset_id >> dataset_id; }
	while (!ifs_core_no.eof()) { ifs_core_no >> core_no; }
	while (!ifs_sampling_rate.eof()) { ifs_sampling_rate >> sampling_rate; }

	// determine delta_min & rho_min
	if (dataset_id == 0) {

		delta_min = 10000;
		local_density_min = 5;
	}
}

// data input
void input_data() {

	// point variable
	point p;
	pt p_;

	// id variable
	unsigned int id = 0;

	// position & id variables
	std::vector<float> d(dimensionality);
	std::vector<float> d_max(dimensionality);
	std::vector<float> d_min(dimensionality);
	for (unsigned int i = 0; i < dimensionality; ++i) d_min[i] = 1000000;

	// sample probablility
	std::mt19937 mt(1);
	std::uniform_real_distribution<> rnd(0, 1.0);

	// dataset for visual check
	if (dataset_id == 0) {

		// position & id variables
		float x = 0, y = 0;

		std::string f_name = "../_dataset/synthetic.txt";

		// file input
		std::ifstream ifs_file(f_name);

		// error check
		if (ifs_file.fail()) {
			std::cout << " data file does not exist." << std::endl;
			std::exit(0);
		}

		// file read
		while (!ifs_file.eof()) {

			// input
			ifs_file >> x >> y;

			p[0] = x;
			p[1] = y;

			if (x > d_max[0]) d_max[0] = x;
			if (x < d_min[0]) d_min[0] = x;
			if (y > d_max[1]) d_max[1] = y;
			if (y < d_min[1]) d_min[1] = y;

			// update pt
			p_.update_id(id);
			p_.update_val(p);

			// insert into dataset
			dataset_pt.push_back(p_);

			// increment identifier
			++id;
		}
	}

	// normalization if necessary
	float coord_max = 100000;
	bool flag = 1;
	if (flag) {

		for (unsigned int i = 0; i < dimensionality; ++i) d_max[i] -= d_min[i];

		for (unsigned int i = 0; i < dataset_pt.size(); ++i) {
			for (unsigned int j = 0; j < dimensionality; ++j) {
				dataset_pt[i][j] -= d_min[j];
				dataset_pt[i][j] /= d_max[j];
				dataset_pt[i][j] *= coord_max;
			}
		}
	}

	// reserve memory
	rnn.reserve(dataset_pt.size());
}

// compute directory
std::string directory_comp() {

	if (dataset_id == 0) return "0-synthetic";

	return "";
}

// output decision graph
void output_decision_graph() {

	std::string f_name = "result/" + directory_comp() + "/Decision-graph/data-id(" + std::to_string(dataset_id) + ")_cutoff(" + std::to_string(cutoff) + ").csv";
	std::ofstream file;
	file.open(f_name.c_str(), std::ios::out | std::ios::app);

	if (file.fail()) {
		std::cerr << " cannot open the output file." << std::endl;
		file.clear();
		return;
	}

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) {
		if (dataset_pt[i].local_density >= local_density_min) file << dataset_pt[i].id << "," << dataset_pt[i].local_density << "," << dataset_pt[i].NN_dist << "\n";
	}

	file.close();
}

// output label
void output_label() {

	std::string f_name = "result/" + directory_comp() + "/Label/label_id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ").txt";
	std::ofstream file;
	file.open(f_name.c_str(), std::ios::out | std::ios::app);

	if (file.fail()) {
		std::cerr << " cannot open the output file." << std::endl;
		file.clear();
		return;
	}

	for (unsigned int i = 0; i < dataset_pt.size(); ++i) file << dataset_pt[i].id << "\t" << dataset_pt[i].label << "\n";

	file.close();
}

// output label (only labels as a list)
void output_label_() {
	std::string f_name = "result/" + directory_comp() + "/Label/lab_id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ").txt";
	
	std::ofstream file;
	file.open(f_name.c_str(), std::ios::out | std::ios::app);

	if (file.fail()) {
		std::cerr << " cannot open the output file." << std::endl;
		file.clear();
		return;
	}

	std::sort(dataset_pt.begin(), dataset_pt.end(), asc_id);
	for (unsigned int i = 0; i < dataset_pt.size(); ++i) file << dataset_pt[i].label << "\n";

	file.close();
}

// output computation time
void output_cpu_time() {

	std::string f_name = "result/" + directory_comp() + "/cpu_time_data-id(" + std::to_string(dataset_id) + ")_sampling_rate(" + std::to_string(sampling_rate) + ")_cutoff(" + std::to_string(cutoff) + ")_core_no(" + std::to_string(core_no) + ").csv";
	std::ofstream file;
	file.open(f_name.c_str(), std::ios::out | std::ios::app);

	if (file.fail()) {
		std::cerr << " cannot open the output file." << std::endl;
		file.clear();
		return;
	}

	file <<
		"Pre-porcessing time [microsec]" << "," <<
		"Local density comp. time [microsec]" << "," <<
		"Dependensy comp. time [microsec]" << "," <<
		"Label propagation time [microsec]" << "," <<
		"Total time [microsec]" << "," <<
		"Number of clusters" << "," <<
		"Noise rate" << "," <<
		"\n";
	file << 
		cpu_offline << "," << 
		cpu_local_density << "," << 
		cpu_dependency << "," << 
		cpu_label << "," << 
		cpu_dependency + cpu_local_density + cpu_label <<  "," << 
		cluster_centers.size() << "," << 
		"\n\n";

	file.close();
}
