#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "libs/argparse.hpp"

const std::string __version__ = "0.1";
const size_t seed = std::chrono::system_clock::now().time_since_epoch().count();

struct Data {
  Data(int size) : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
  }
  Data(int size, std::vector<float> &hx, std::vector<float> &hy)
  : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemcpy(x, hx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy.data(), bytes, cudaMemcpyHostToDevice);
  }
  ~Data() {
    cudaFree(x);
    cudaFree(y);
  }
  float *x{nullptr};
  float *y{nullptr};
  int size{0};
  int bytes{0};
};

__device__ float norm(float x_one, float y_one, float x_two, float y_two) {
  return (x_one - x_two) * (x_one - x_two) + (y_one - y_two) * (y_one - y_two);
}

__global__ void fine_reduce(
  const float* __restrict__ x_data,
  const float* __restrict__ y_data,
  int data_size,
  const float* __restrict__ x_means,
  const float* __restrict__ y_means,
  float* __restrict__ x_new_sums,
  float* __restrict__ y_new_sums,
  int num_clusters,
  int* __restrict__ counts
) {
  extern __shared__ float shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size) return;

  if (local_index < num_clusters) {
    shared_data[local_index] = x_means[local_index];
    shared_data[local_index + num_clusters] = y_means[local_index];
  }

  __syncthreads();

  const float x_value = x_data[global_index];
  const float y_value = y_data[global_index];

  float best_distance = FLT_MAX;
  size_t best_cluster = 0;

  for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
    const float distance = norm(
      x_value, y_value,
      shared_data[cluster],
      shared_data[cluster + num_clusters]
    );
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  __syncthreads();

  const int x = local_index;
  const int y = local_index + 1 * blockDim.x;
  const int z = local_index + 2 * blockDim.x;

  for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
    shared_data[x] = (best_cluster == cluster) ? x_value : 0;
    shared_data[y] = (best_cluster == cluster) ? y_value : 0;
    shared_data[z] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_index < stride) {
        shared_data[x] += shared_data[x + stride];
        shared_data[y] += shared_data[y + stride];
        shared_data[z] += shared_data[z + stride];
      }
      __syncthreads();
    }

    if (local_index == 0) {
      const int cluster_index = blockIdx.x * num_clusters + cluster;
      x_new_sums[cluster_index] = shared_data[x];
      y_new_sums[cluster_index] = shared_data[y];
      counts[cluster_index] = shared_data[z];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(
  float* __restrict__ x_means,
  float* __restrict__ y_means,
  float* __restrict__ x_new_sum,
  float* __restrict__ y_new_sum,
  int num_clusters,
  int* __restrict__ counts
) {
  extern __shared__ float shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  shared_data[index] = x_new_sum[index];
  shared_data[index + y_offset] = y_new_sum[index];
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride >= num_clusters; stride /= 2) {
    if (index < stride) {
      shared_data[index] += shared_data[index + stride];
      shared_data[index + y_offset] += shared_data[index + stride + y_offset];
    }
    __syncthreads();
  }

  if (index < num_clusters) {
    const int count = max(counts[index], 1);
    x_means[index] = x_new_sum[index] / count;
    y_means[index] = y_new_sum[index] / count;
    x_new_sum[index] = 0;
    y_new_sum[index] = 0;
    counts[index] = 0;
  }
}

std::pair<std::vector<float>, std::vector<float>> load_data(const std::string &data_file) {
  std::ifstream file(data_file);
  if (!file) {
    std::cerr << data_file << ": " << std::strerror(errno) << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  std::pair<std::vector<float>, std::vector<float>> data;
  while (std::getline(file, line)) {
    size_t comma = line.find(",");
    float x_data = std::stof(line.substr(0, comma));
    float y_data = std::stof(line.substr(comma + 1, line.length()));
    data.first.push_back(x_data);
    data.second.push_back(y_data);
  }
  file.close();
  return data;
}

void dump_means(const std::vector<float> &x_means, const std::vector<float> &y_means) {
  for (size_t cluster = 0; cluster < x_means.size(); ++cluster)
    std::cout << x_means[cluster] << "," << y_means[cluster] << "\n";
}

int main(int argc, char **argv) {
  argparse::ArgumentParser parser(argv[0], __version__, argparse::default_arguments::none);

  parser
    .add_description("Implements kmeans clustering in CUDA.");
  parser
    .add_epilog("Author: Filemon Mateus (mateus@utah.edu).");
  parser
    .add_argument("-h", "--help")
    .action([&](const auto& /* unused */) {
      std::cout << parser.help().str();
      std::exit(EXIT_SUCCESS);
    })
    .help("Show this help message and exit.")
    .flag();
  parser
    .add_argument("-v", "--version")
    .action([&](const auto& /* ununsed */) {
      std::cout << argv[0] << " " << __version__ << std::endl;
      std::exit(EXIT_SUCCESS);
    })
    .help("Prints version information and exits.")
    .flag();
  parser
    .add_argument("--data_file")
    .help("Path to the input data file containing the data to be clustered.")
    .required()
    .nargs(1);
  parser
    .add_argument("--num_clusters")
    .help("Number of clusters to generate.")
    .required()
    .nargs(1)
    .scan<'i', int>();
  parser
    .add_argument("--num_threads")
    .help("Number of GPU threads to run the clustering subroutine with.")
    .default_value(1024)
    .nargs(1)
    .scan<'i', int>();
  parser
    .add_argument("--max_iter")
    .help("Maximum number of iterations.")
    .default_value(300)
    .nargs(1)
    .scan<'i', int>();

  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto data_file = parser.get<std::string>("--data_file");
  auto num_clusters = parser.get<int>("--num_clusters");
  auto threads = parser.get<int>("--num_threads");
  auto max_iter = parser.get<int>("--max_iter");
  auto data = load_data(data_file);

  std::vector<float> hx = data.first;
  std::vector<float> hy = data.second;

  const size_t num_points = hx.size();

  Data d_data(num_points, hx, hy);

  std::default_random_engine seeder(seed);
  std::mt19937_64 generator(seeder());
  std::shuffle(hx.begin(), hx.end(), generator);
  std::shuffle(hy.begin(), hy.end(), generator);

  const int blocks = (num_points + threads - 1) / threads;
  const int fine_shared_memory = num_clusters * threads * sizeof(float);
  const int coarse_shared_memory = 2 * num_clusters * blocks * sizeof(float);

  Data d_means(num_clusters, hx, hy);
  Data d_sigma(num_clusters * blocks);

  int *d_counts;
  cudaMalloc(&d_counts, num_clusters * blocks * sizeof(int));
  cudaMemset(d_counts, 0, num_clusters * blocks * sizeof(int));

  const auto tic = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < max_iter; ++iteration) {
    fine_reduce<<<blocks, threads, fine_shared_memory>>>(
      d_data.x,
      d_data.y,
      d_data.size,
      d_means.x,
      d_means.y,
      d_sigma.x,
      d_sigma.y,
      num_clusters,
      d_counts
    );
    cudaDeviceSynchronize();
    coarse_reduce<<<1, num_clusters * blocks, coarse_shared_memory>>>(
      d_means.x,
      d_means.y,
      d_sigma.x,
      d_sigma.y,
      num_clusters,
      d_counts
    );
    cudaDeviceSynchronize();
  }
  const auto toc = std::chrono::high_resolution_clock::now();

  cudaFree(d_counts);

  std::vector<float> x_means(num_clusters, 0);
  std::vector<float> y_means(num_clusters, 0);
  cudaMemcpy(x_means.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(y_means.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);

  dump_means(x_means, y_means);

  double total_time = std::chrono::duration<double>(toc-tic).count();
  std::cout << "# runtime: " << std::setprecision(6) << std::fixed << total_time << std::endl;
  return 0;
}
