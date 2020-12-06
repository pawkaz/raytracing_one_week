#include <iostream>
#include <fstream>


int main() {
  int nx = 200;
  int ny = 100;

  std::ofstream myfile;
  myfile.open ("test.ppm");


  myfile << "P3\n" << nx << " " << ny << "\n255\n";

  for (int j = ny - 1; j >= 0; --j) {
    for (int i = 0; i < nx; ++i) {
        float r = float(i) / nx;
        float g = float(j) / ny;
        float b = 0.2;

        int ir = int(255.99 * r);
        int ig = int(255.99 * g);
        int ib = int(255.99 * b);

        myfile << ir << " " << ig << " " << ib << "\n";
    }
  }
  myfile.close();
  return 0;
}