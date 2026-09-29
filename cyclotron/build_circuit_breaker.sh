# Build test binary
g++ -std=c++20 test_circuit_breaker.cpp -lgtest -lgtest_main -pthread -o test_circuit_breaker

# Run test suite
./test_circuit_breaker