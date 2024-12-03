# List-Application-on-ANN
This is the implementation of Dense List, Doubly Linked List and their application on creating a dataloader for training models. My implementation uses the available library xtensor of C++ language. The zip contains the entire project with testcases.

The testcases related to dataloader need some configuration on the random seed.

Command to build and run:
Build: g++ -fsanitize=address  -o main -Iinclude -Isrc -std=c++17 main.cpp test/unit_test/arrayList/unit_test.cpp test/unit_test/linkedList/unit_test.cpp test/unit_test/data/unit_test.cpp test/random_test/random_test.cpp src/ann/xtensor_lib.cpp src/util/Point.cpp
Run:
* terminal unit test array list
    !./main test_unit_array
    !./main test_unit_array nameFunctionUnitTest

    * terminal unit test linked list
    ! ./main test_unit_linkedList
    ! ./main test_unit_linkedList nameFunctionUnitTest

    * terminal unit test dataset
    ! ./main test_unit_data
    ! ./main test_unit_data nameFunctionUnitTest

    * terminal auto test
    ! ./main test_random number_1 number_2
    ! ./main test_random number
