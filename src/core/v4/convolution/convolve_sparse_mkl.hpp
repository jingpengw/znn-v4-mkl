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

    size_t rx = ax - (bx-1)*s[0];
    size_t ry = ay - (by-1)*s[1];
    size_t rz = az - (bz-1)*s[2];

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    // 3d convolution using MKL
    // temporal volume size
    MKL_INT tashape[3]={(az-1)/s[2]+1, (ay-1)/s[1]+1, (ax-1)/s[0]+1};
    MKL_INT trshape[3]={tashape[0]-bz+1, tashape[1]-by+1, tashape[2]-bx+1};

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
                MKL_INT trshape[3]={tashape[0]-bz+1, tashape[1]-by+1, tashape[2]-bx+1};

                // prepare input
                for (std::size_t x=xs, xt=0; x<ax; x+=s[0], xt++)
                    for (std::size_t y=ys, yt=0; y<ay; y+=s[1], yt++)
                        for(std::size_t z=zs, zt=0; z<az; z+=s[2], zt++)
                            ta[ zt+ yt*tashape[0] + xt*tashape[1]*tashape[0] ] = a[x][y][z];

                //std::cout<<"status-->conv delete task:  "<<status<<std::endl;
                int status = vsldConvExec(conv_plans.get(vec3i(tashape[0], tashape[1], tashape[2]), vec3i(bz,by,bx)),
                                          ta, NULL,
                                          b.data(), NULL,
                                          tr, NULL);

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
