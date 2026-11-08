__kernel void v_multiply(__global const float *a,
                         __global const float *b,
                         __global float *result)
{
    int global_id = get_global_id(0);
    result[global_id] = a[global_id] * b[global_id];
}

// EOF
