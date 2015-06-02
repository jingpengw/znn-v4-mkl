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

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

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

    // temporal volume size
    MKL_INT tashape[3]={(az-1)/s[2]+1, (ay-1)/s[1]+1, (ax-1)/s[0]+1};
    MKL_INT trshape[3]={tashape[0]-tbshape[0]+1, tashape[1]-tbshape[1]+1, tashape[2]-tbshape[2]+1};

    // temporal subconvolution output
    double ta[ tashape[0]* tashape[1]* tashape[2] ];
    double tr[ trshape[0]* trshape[1]* trshape[2] ];

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                // temporal volume size
                MKL_INT tashape[3]={(az-zs-1)/s[2]+1, (ay-ys-1)/s[1]+1, (ax-xs-1)/s[0]+1};
                MKL_INT trshape[3]={tashape[0]-tbshape[0]+1, tashape[1]-tbshape[1]+1, tashape[2]-tbshape[2]+1};

                // prepare input
                for (std::size_t x=xs, xt=0; x<ax; x+=s[0], xt++)
                    for (std::size_t y=ys, yt=0; y<ay; y+=s[1], yt++)
                        for(std::size_t z=zs, zt=0; z<az; z+=s[2], zt++)
                            ta[ zt+ yt*tashape[0] + xt*tashape[1]*tashape[0] ] = a[x][y][z];

                // subconvolution
                //std::cout<<"subconvolution..."<<std::endl;
                status = vsldConvNewTask(&task,mode,dims,tashape, tbshape, trshape);
                //std::cout<<"status-->new task:          "<<status<<std::endl;
                status = vslConvSetStart(task, start);
                //std::cout<<"status-->set start:         "<<status<<std::endl;
                status = vsldConvExec(task, ta, NULL, b.data(), NULL, tr, NULL);
                //std::cout<<"status-->conv exec:         "<<status<<std::endl;
                status = vslConvDeleteTask(&task);
                //std::cout<<"status-->conv delete task:  "<<status<<std::endl;

                // combine subconvolution results
                for (std::size_t x=xs, wx=0; x<rx; x+=s[0], wx++)
                    for (std::size_t y=ys, wy=0; y<ry; y+=s[1], wy++ )
                        for (std::size_t z=zs, wz=0; z<rz; z+=s[2], wz++)
                            r[x][y][z] = tr[wz + wy*trshape[0] + wx*trshape[1]*trshape[0] ];
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
