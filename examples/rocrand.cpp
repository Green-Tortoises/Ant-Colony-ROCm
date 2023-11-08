#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

__global__
void test() {
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	rocrand_state_xorwow state;
	rocrand_init(123, tid, 0, &state);

	for (int i = 0; i < 3; ++i) {
		const auto value = rocrand(&state);
		printf("thread %d, index %u: %u\n", tid, i, value);
	}
}

int main() {
	test<<<dim3(1), dim3(32)>>>();
	hipDeviceSynchronize();
}
