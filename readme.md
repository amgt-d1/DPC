## Introduction
* This repository provides implementations of Ex-DPC, Approx-DPC, and S-Approx-DPC.
* They are fast algorithms for [density-peaks clustering](https://science.sciencemag.org/content/344/6191/1492.full) (proposed in Science).
* As for the details about these algorithms, please read our SIGMOD2021 paper, [Fast Density-Peaks Clustering: Multicore-based Parallelization Approach](https://).

## Requirement
*  [spatial liberary](http://spatial.sourceforge.net/)
	* We used version 2.1.8.
* Boost 1.67.0
	* We have not confirmed the availability of the other version.
* The source codes of DPC algorithms have to be changed based on your paths of the above libraries.

## How to use
* We prepared codes for Windows OS (Visual Studio) and Linux (Ubuntu).
* Approx-DPC is available only for Windows OS.
	* This is because of the usage of `concurrent_vector.h` and `concurrent_unordered_map.h`.
	* For Linux-based OS, the above functions can be replaced by `#pragma omp critical` (but the above functions are faster in our environment).
* We assume low-dimensional datasets, as we use a kd-tree.
	* Implementation is based on https://github.com/gishi523/kd-tree and spatial library.

### Visual Studio 2019 (64-bit)
* Make a new project -> console application.
* Put our source codes into the project as with `dataset`, `parameter`, and `result` folders.
* Include the paths to `spatial library` and `boost library`.
* Enable `OpenMP`. 
	* You may need `/Zc:twoPhase-` as an additional option at command line of C/C++ in property.
* To complie, use `/Ox` option.
* For Approx-DPC, set NO as `SDL check`.

### Linux
* Compile: `g++ -O3 Ex-DPC.cpp -o exdpc.out -fopenmp` and run: `./exdpc.out`.
* Compile: `g++ -O3 S-Approx-DPC.cpp -o sapproxdpc.out -fopenmp` and run: `./sapproxdpc.out`.

### Datasets
* As an example, we have prepared a 2-dimensional synthetic dataset used in our paper.
* If you want to test your dataset,
	* Put the file at `_dataset` or `dataset` directory.
	* Assign a unique dataset ID.
	* Set the dimensionality at `data.hpp`.
	* Write codes for inputing the data file in `input_data()` function of `file_io.hpp`.  
	* Add a direcotry in `result` and update the function `compute_direcotry()`.
	* Compile the code and run .exe or .out file.

### Parameters
* Set some value in the corresponding txt file in `parameter` or `_parameter`.
* For \rho_min and \delta_min, we specify them in `file_io.hpp`.

### Ex-DPC
* Uncomment line 211 if you need cluster labels of the exact answer.

### Approximation algorithms
* If you want to compute rand index, you have to run Ex-DPC and obtain the cluster labels.

## Citation
If you use our implementation, please cite the following paper.
``` 
@inproceedings{amagata2021dpc,  
    title={Fast Density-Peaks Clustering: Multicore-based Parallelization Approach},  
    author={Amagata, Daichi and Hara, Takahiro},  
    booktitle={SIGMOD},  
    pages={xxx--xxx},  
    year={2021}  
}
```

## License
Copyright (c) 2020 Daichi Amagata  
This software is released under the [MIT license](https://github.com/amgt-d1/DPC/blob/main/license.txt).
