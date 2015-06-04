#pragma once

#include "../types.hpp"
#include "../cube/cube_operators.hpp"
#include "convolve_constant.hpp"
#include <zi/utility/singleton.hpp>
#include <mkl_vsl.h>

#include "convolve_mkl.hpp"

namespace znn { namespace v4 {

template< typename T >
inline void convolve_sparse_add_mkl(    cube<T> const   & a,
                                        cube<T> const   & b,
                                        vec3i const     & s,
                                        cube<T>         & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_mkl(a,b,r);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];
    std::cout<<"filter size (z,y,x): "<<bz<<", "<<by<<", "<<bx<<std::endl;

    size_t rx = ax - (bx-1)*s[0];
    size_t ry = ay - (by-1)*s[1];
    size_t rz = az - (bz-1)*s[2];

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    // 3d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=3;
    int status;
    const int mode = VSL_CONV_MODE_DIRECT;//direct convolution--DIRECT, FFT

    // the kernel
    MKL_INT tbshape[3]={bz, by, bx};

    int start[3]={tbshape[0]-1, tbshape[1]-1,tbshape[2]-1};
    //std::cout<< "start: "<<start[0]<<", "<<start[1]<<", "<<start[2]<<std::endl;

    // stride
    const MKL_INT strides_in[3]  = { s[2], az*s[1], az*ay*s[0] };
    const MKL_INT strides_out[3] = { s[2], rz*s[1], rz*ry*s[0] };
    std::cout<<"stride in : "<< strides_in[0]  << "," << strides_in[1]  << "," << strides_in[2]  <<std::endl;
    std::cout<<"stride out: "<< strides_out[0] << "," << strides_out[1] << "," << strides_out[2] <<std::endl;

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                // temporal volume size
                MKL_INT tashape[3]={(az-zs-1)/s[2]+1, (ay-ys-1)/s[1]+1, (ax-xs-1)/s[0]+1};
                // MKL_INT trshape[3]={tashape[0]-tbshape[0]+1, tashape[1]-tbshape[1]+1, tashape[2]-tbshape[2]+1};
                MKL_INT trshape[3]={rz, ry, rx};

                const T* in_ptr  = &(a[xs][ys][zs]);
                T* out_ptr = &(r[xs][ys][zs]);

                // subconvolution
                //std::cout<<"subconvolution..."<<std::endl;
                status = vsldConvNewTask(&task,mode,dims,tashape, tbshape, trshape);
                //std::cout<<"status-->new task:          "<<status<<std::endl;
                status = vslConvSetStart(task, start);
                //std::cout<<"status-->set start:         "<<status<<std::endl;
                status = vsldConvExec(task, in_ptr, strides_in, b.data(), NULL, out_ptr, strides_out);
                std::cout<<"status-->conv exec:         "<<status<<std::endl;
                status = vslConvDeleteTask(&task);
                std::cout<<"status-->conv delete task:  "<<status<<std::endl;
            }
}

template< typename T >
inline void convolve_sparse_mkl( cube<T> const & a,
                             cube<T> const & b,
                             vec3i const & s,
                             cube<T> & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_mkl(a,b,r);
        return;
    }

    fill(r,0);
    convolve_sparse_add_mkl(a,b,s,r);
}


template< typename T >
inline cube_p<T> convolve_sparse_mkl( cube<T> const & a,
                                  cube<T> const & b,
                                  vec3i const & s )
{
    if ( s == vec3i::one )
    {
        return convolve_mkl(a,b);
    }

    cube_p<T> r = get_cube<T>(size(a) - (size(b) - vec3i::one) * s);

    fill(*r,0);
    convolve_sparse_add_mkl(a,b,s,*r);
    return r;
}

template< typename T >
inline cube_p<T> convolve_sparse_mkl( ccube_p<T> const & a,
                                  ccube_p<T> const & b,
                                  vec3i const & s )
{
    return convolve_sparse_mkl(*a,*b,s);
}


}} // namespace znn::v4
