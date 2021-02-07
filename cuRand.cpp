#include <iostream>
#include <iostream>
#include <string>
#include <random>
#include <ctime>

int main()
{
  //Variables declaration
  double v1[10];
  double normalized[10];

  //Normal distribution
  std::random_device mch;
  std::default_random_engine generator(mch()); //Seeding
  std::normal_distribution<double> distribution(0.0, 1.0); //Parameters are: Distribution mean, Standard deviation
  for(int i=1; i<=10; i++)
  {
    normalized[i] = distribution(generator);
    printf("normal_distribution (0.0,1.0): %f\n", normalized[i]);
  }

//Basic random distribution
  srand((unsigned int)time(NULL)); //seeding
  for (int i=1; i<=10; i++)
  {
    v1[i] = (double)rand() / ( RAND_MAX / (100-1) ) ; //Random float number in range [1,100]
    printf("Basic random double number: %f\n", v1 [i]);
  }
  return 0;
}
