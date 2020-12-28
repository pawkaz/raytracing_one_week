
#include <chrono>
#include <fstream>
#include <iostream>
// #include <opencv2/opencv.hpp>


// class vec3;

void render_image(float* fb, unsigned nx, unsigned ny);

int main() {
  using namespace std;
  using namespace std::chrono;
  using clock = std::chrono::high_resolution_clock;

  std::cout << "Hello world" << std::endl;
  const auto aspect_ratio = 16.0 / 9.0;
  const unsigned nx = 1920;
  const unsigned ny = static_cast<int>(nx / aspect_ratio);
  
  auto start = clock::now();

  // allocate FB
  auto* h_fb = new float[nx * ny * 3];
  render_image(h_fb, nx, ny);

  std::cout << "Ended in " << duration_cast<seconds>(clock::now() - start).count() << "ms\n";

  ofstream file;
  file.open("test.ppm");

  file << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t idx = j * 3 * nx + i * 3;
      float r = h_fb[idx + 0];
      float g = h_fb[idx + 1];
      float b = h_fb[idx + 2];
      int ir = int(255.99 * r);
      int ig = int(255.99 * g);
      int ib = int(255.99 * b);
      file << ir << " " << ig << " " << ib << "\n";
    }
  }
  file.close();

  // cv::Mat img = cv::Mat(ny, nx, CV_32FC3, h_fb);
  // cv::imwrite("test.jpg", img);
  delete[] h_fb;

  return 0;
}