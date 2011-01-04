import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

import numpy

cross4_preamble = """
    float distance2(float4 a,
            float4 b)
    {   
        return (a.x*a.x - b.x*b.x) + 
            (a.y*a.y - b.y*b.y) +
            (a.z*a.z - b.z*b.z) +
            (a.w*a.w - b.w*b.w);
    }

    float4 cross4(float4 u,
            float4 v,
            float4 w)
    {   
        int i = get_global_id(0);
        float4 r;
        float A = (v.x * w.y) - (v.y * w.x);
        float B = (v.x * w.z) - (v.z * w.x);
        float C = (v.x * w.w) - (v.w * w.x);
        float D = (v.y * w.z) - (v.z * w.y);
        float E = (v.y * w.w) - (v.w * w.y);
        float F = (v.z * w.w) - (v.w * w.z);

        r.x =   (u.y * F) - (u.z * E) + (u.w * D);
        r.y = - (u.x * F) + (u.z * C) - (u.w * B);
        r.z =   (u.x * E) - (u.y * C) + (u.w * A);
        r.w = - (u.x * D) + (u.y * B) - (u.z * A);

        return r;
    }
"""


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

cross4 = ElementwiseKernel(ctx,
        "__global const float4 *u, "
        "__global const float4 *v, "
        "__global const float4 *w, "
        "__global       float4 *r",
        "r[i] = cross4(u[i],v[i],w[i])",
        "cross4_final", preamble=cross4_preamble)


a = numpy.random.randn(4,4).astype(numpy.float32)
b = numpy.random.randn(4,4).astype(numpy.float32)
c = numpy.random.randn(4,4).astype(numpy.float32)

a_gpu = cl_array.to_device(ctx, queue, a)
b_gpu = cl_array.to_device(ctx, queue, b)
c_gpu = cl_array.to_device(ctx, queue, c)

dest_gpu = cl_array.empty_like(a_gpu)

def main():
    pass

if __name__ == "__main__":
    main()
