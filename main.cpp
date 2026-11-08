#include <iostream>

#define CL_TARGET_OPENCL_VERSION 310
#include <CL/cl.h>

bool checkResult(cl_int retval, const std::string &message)
{
    if (retval != CL_SUCCESS)
    {
        std::cerr << "ERROR: " << message << std::endl
                  << "Code: " << retval << std::endl;
        return false;
    }

    return true;
}

int main()
{
    std::cout << "Hello world!" << std::endl;

    constexpr size_t vec_size = 4;
    constexpr size_t vec_size_bytes = vec_size * sizeof(float);
    float vec_a[vec_size] = {1., 2., 3., 4.};
    float vec_b[vec_size] = {5., 6., 7., 8.};
    float vec_result[vec_size] = {0., 0., 0., 0.};

    // OpenCL variables.
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_v_multiply;

    // Initialise OpenCL program.
    cl_uint num_platforms;
    if (!checkResult(clGetPlatformIDs(0, NULL, &num_platforms),
                     "Could not get plaform IDs"))
    {
        return EXIT_FAILURE;
    }

    if (num_platforms == 0)
    {
        std::cerr << "No platforms found!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Num platforms: " << num_platforms << std::endl;

    cl_platform_id *platform_ids = new cl_platform_id[num_platforms];
    if (!checkResult(clGetPlatformIDs(num_platforms, platform_ids, NULL),
                     "Could not get plaform details"))
    {
        return EXIT_FAILURE;
    }

    cl_device_id * device_ids = new cl_device_id[num_platforms];
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
        if (!checkResult(clGetDeviceIDs(platform_ids[i],
                                        CL_DEVICE_TYPE_ALL, 1,
                                        &device_ids[i], NULL),
                         "Could not get device IDs"))
        {
            return EXIT_FAILURE;
        }

        std::cout << "  [" << i+1 << "]: ";
        char device_name[1024];
        if (checkResult(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME,
                                        sizeof(device_name) / sizeof(char),
                                        device_name, NULL),
                        "Could not get device name"))
        {
            std::cout << device_name << std::endl;
        }
        else
        {
            std::cout << "<unknown>" << std::endl;
        }
    }

    cl_uint device_num = (num_platforms == 1 ? 1 : 0);
    while (device_num <= 0 || device_num > num_platforms)
    {
        std::cout << "Select device [1-" << num_platforms << "]: ";
        std::cin >> device_num;
    }

    std::cout << "Using device " << device_num << std::endl;

    cl_device_id device_id = device_ids[device_num - 1];

    // Set up OpenCL for this device.
    cl_int retval;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &retval);
    if (!checkResult(retval, "Could not create context"))
    {
        return EXIT_FAILURE;
    }

    queue = clCreateCommandQueueWithProperties(context, device_id,
                                               0, &retval);
    if (!checkResult(retval, "Could not create queue"))
    {
        return EXIT_FAILURE;
    }


    const char kernel_source[] = {
        #embed "kernel.cl"
    };
    const char *kernel_ptr = kernel_source; // pointer decay
    program = clCreateProgramWithSource(context, 1, &kernel_ptr, NULL, &retval);
    if (!checkResult(retval, "Could not create program"))
    {
        return EXIT_FAILURE;
    }

    // Build the program.
    if (!checkResult(clBuildProgram(program, 0, NULL, NULL, NULL, NULL),
                     "Could not build program"))
    {
        // Build failed - show the compilation error.
        size_t len;
        char buffer[4096];

        if (checkResult(clGetProgramBuildInfo(program, device_id,
                                              CL_PROGRAM_BUILD_LOG,
                                              sizeof(buffer) / sizeof(char),
                                              buffer, &len),
                        "Could not get build error message"))
        {
            std::cout << std::endl
                      << buffer << std::endl;
        }

        return EXIT_FAILURE;
    }

    kernel_v_multiply = clCreateKernel(program, "v_multiply", &retval);
    if (!checkResult(retval, "Could not create kernel")) {
        return EXIT_FAILURE;
    }

    // Create the memory buffers.
    cl_mem mem_a = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY,
                                  vec_size_bytes, NULL, &retval);
    if (!checkResult(retval, "Could not create memory for vector a")) {
        return EXIT_FAILURE;
    }

    cl_mem mem_b = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY,
                                  vec_size_bytes, NULL, &retval);
    if (!checkResult(retval, "Could not create memory for vector b")) {
        return EXIT_FAILURE;
    }

    cl_mem mem_result = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY,
                                       vec_size_bytes, NULL, &retval);
    if (!checkResult(retval, "Could not create memory for vector result")) {
        return EXIT_FAILURE;
    }

    // Add the arrays as arguments to the OpenCL function.
    retval  = clSetKernelArg(kernel_v_multiply, 0, sizeof(cl_mem), &mem_a);
    retval |= clSetKernelArg(kernel_v_multiply, 1, sizeof(cl_mem), &mem_b);
    retval |= clSetKernelArg(kernel_v_multiply, 2, sizeof(cl_mem), &mem_result);
    if (!checkResult(retval, "Could not set kernel arguments")) {
        return EXIT_FAILURE;
    }

    // Copy the input data to the device (GPU).
    retval  = clEnqueueWriteBuffer(queue, mem_a, CL_TRUE, 0, vec_size_bytes,
                                   vec_a, 0, NULL, NULL);
    retval |= clEnqueueWriteBuffer(queue, mem_b, CL_TRUE, 0, vec_size_bytes,
                                   vec_b, 0, NULL, NULL);
    if (!checkResult(retval, "Could not copy data to device")) {
        return EXIT_FAILURE;
    }

    // Run the kernel and check the results!
    if (!checkResult(clEnqueueNDRangeKernel(queue, kernel_v_multiply,
                                            1, NULL, &vec_size,
                                            NULL, 0, NULL, NULL),
                     "Could not enqueue kernel"))
    {
        return EXIT_FAILURE;
    };
    clFinish(queue);

    if (!checkResult(clEnqueueReadBuffer(queue, mem_result, CL_TRUE, 0,
                                         vec_size_bytes, vec_result,
                                         0, NULL, NULL),
                     "Could not enqueue read buffer"))
    {
        return EXIT_FAILURE;
    }

    for (cl_uint i = 0; i < vec_size; ++i)
    {
        std::cout << vec_a[i] << " x " << vec_b[i] << " = " << vec_result[i]
                  << std::endl;
    }

    // Cleanup.
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_result);
    clReleaseProgram(program);
    clReleaseKernel(kernel_v_multiply);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // FIXME RAII this.
    delete[] platform_ids;
    delete[] device_ids;

    return EXIT_SUCCESS;
}

// EOF
