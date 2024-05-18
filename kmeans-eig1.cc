#include <chrono>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "libs/argparse.hpp"

const std::string __version__ = "0.1";
const size_t seed = std::chrono::system_clock::now().time_since_epoch().count();

Eigen::ArrayXXf kmeans(const Eigen::ArrayXXf &data, int num_clusters, int max_iter) {
  static std::default_random_engine seeder(seed);
  static std::mt19937_64 generator(seeder());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  Eigen::ArrayX2f means(num_clusters, 2);
  for (size_t cluster = 0; cluster < num_clusters; ++cluster)
    means.row(cluster) = data(indices(generator));

  const Eigen::ArrayXXf x_data = data.col(0).rowwise().replicate(num_clusters);
  const Eigen::ArrayXXf y_data = data.col(1).rowwise().replicate(num_clusters);

  for (size_t iteration = 0; iteration < max_iter; ++iteration) {
    Eigen::ArrayXXf distances =
      (x_data.rowwise() - means.col(0).transpose()).square() +
      (y_data.rowwise() - means.col(1).transpose()).square();

    Eigen::ArrayX2f sigma = Eigen::ArrayX2f::Zero(num_clusters, 2);
    Eigen::ArrayXf counts = Eigen::ArrayXf::Ones(num_clusters);

    for (size_t index = 0; index < data.rows(); ++index) {
      Eigen::ArrayXf::Index argmin;
      distances.row(index).minCoeff(&argmin);
      sigma.row(argmin) += data.row(index).array();
      counts(argmin) += 1;
    }

    means = sigma.colwise() / counts;
  }
  return means;
}

Eigen::ArrayXXf load_data(const std::string& data_file) {
  std::ifstream file(data_file);
  if (!file) {
    std::cerr << data_file << ": " << std::strerror(errno) << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  std::vector<std::pair<float, float>> data_temp;
  while (std::getline(file, line)) {
    size_t comma = line.find(",");
    float x_data = std::stof(line.substr(0, comma));
    float y_data = std::stof(line.substr(comma + 1, line.length()));
    data_temp.emplace_back(x_data, y_data);
  }
  Eigen::ArrayXXf data(data_temp.size(), 2);
  for (size_t row = 0; row < data.rows(); ++row)
    data.row(row) << data_temp[row].first, data_temp[row].second;
  file.close();
  return data;
}

void dump_means(const Eigen::ArrayXXf &means) {
  for (size_t cluster = 0; cluster < means.rows(); ++cluster)
    std::cout << means.coeff(cluster, 0) << "," << means.coeff(cluster, 1) << "\n";
}

int main(int argc, char **argv) {
  argparse::ArgumentParser parser(argv[0], __version__, argparse::default_arguments::none);

  parser
    .add_description("Implements kmeans clustering in Eigen.");
  parser
    .add_epilog("Author: Filemon Mateus (mateus@utah.edu).");
  parser
    .add_argument("-h", "--help")
    .action([&](const auto& /* unused */) {
      std::cout << parser.help().str();
      std::exit(EXIT_SUCCESS);
    })
    .help("Show this help message and exits.")
    .flag();
  parser
    .add_argument("-v", "--version")
    .action([&](const auto& /* unused */) {
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
  Eigen::ArrayXXf means;

  const auto tic = std::chrono::high_resolution_clock::now();
  for (size_t trial = 0; trial < num_trials; ++trial)
    means = kmeans(data, num_clusters, max_iter);
  const auto toc = std::chrono::high_resolution_clock::now();

  dump_means(means);

  double total_time = std::chrono::duration<double>(toc-tic).count() / num_trials;
  std::cout << "# runtime: " << std::setprecision(6) << std::fixed << total_time << std::endl;
  return 0;
}
