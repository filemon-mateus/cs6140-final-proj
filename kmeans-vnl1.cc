#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "libs/argparse.hpp"

const std::string __version__ = "0.1";
const size_t seed = std::chrono::system_clock::now().time_since_epoch().count();

struct Point {
  float x{0.f}, y{0.f};
};

using DataFrame = std::vector<Point>;

float square(float value) {
  return value * value;
}

float norm(const Point& p, const Point& q) {
  return square(p.x - q.x) + square(p.y - q.y);
}

DataFrame kmeans(const DataFrame &data, int num_clusters, int max_iter) {
  static std::default_random_engine seeder(seed);
  static std::mt19937_64 generator(seeder());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  DataFrame means(num_clusters);
  for (auto &cluster : means)
    cluster = data[indices(generator)];

  std::vector<size_t> labels(data.size());
  for (size_t iteration = 0; iteration < max_iter; ++iteration) {
    for (size_t point = 0; point < data.size(); ++point) {
      float best_distance = std::numeric_limits<float>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        const float distance = norm(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      labels[point] = best_cluster;
    }

    DataFrame new_means(num_clusters);
    std::vector<size_t> counts(num_clusters, 0);
    for (size_t point = 0; point < data.size(); ++point) {
      const auto cluster = labels[point];
      new_means[cluster].x += data[point].x;
      new_means[cluster].y += data[point].y;
      counts[cluster] += 1;
    }

    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
      // turn 0/0 into 0/1 to avoid zero division
      const auto count = std::max<size_t>(1, counts[cluster]);
      means[cluster].x = new_means[cluster].x / count;
      means[cluster].y = new_means[cluster].y / count;
    }
  }
  return means;
}

DataFrame load_data(const std::string &data_file) {
  std::ifstream file(data_file);
  if (!file) {
    std::cerr << data_file << ": " << std::strerror(errno) << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  DataFrame data;
  while (std::getline(file, line)) {
    size_t comma = line.find(",");
    float x_data = std::stof(line.substr(0, comma));
    float y_data = std::stof(line.substr(comma + 1, line.length()));
    data.push_back(Point{x_data, y_data});
  }
  file.close();
  return data;
}

void dump_means(const DataFrame &means) {
  for (const auto &cluster : means)
    std::cout << cluster.x << "," << cluster.y << "\n";
}

int main(int argc, char **argv) {
  argparse::ArgumentParser parser(argv[0], __version__, argparse::default_arguments::none);

  parser
    .add_description("Implements kmeans clustering in vanilla C++.");
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
    .add_argument("--max_iter")
    .help("Maximum number of iterations.")
    .default_value(300)
    .nargs(1)
    .scan<'i', int>();
  parser
    .add_argument("--num_trials")
    .help("Number of trials to perform clustering subroutine.")
    .default_value(5)
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
  auto max_iter = parser.get<int>("--max_iter");
  auto num_trials = parser.get<int>("--num_trials");
  auto data = load_data(data_file);
  DataFrame means;

  const auto tic = std::chrono::high_resolution_clock::now();
  for (size_t trial = 0; trial < num_trials; ++trial)
    means = kmeans(data, num_clusters, max_iter);
  const auto toc = std::chrono::high_resolution_clock::now();

  dump_means(means);

  double total_time = std::chrono::duration<double>(toc-tic).count() / num_trials;
  std::cout << "# runtime: " << std::setprecision(6) << std::fixed << total_time << std::endl;
  return 0;
}
