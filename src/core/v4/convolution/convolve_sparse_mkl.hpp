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

    const MKL_INT strides_in[3]  = { s[2], az*s[1], az*ay*s[0] };
    const MKL_INT strides_out[3] = { s[2], rz*s[1], rz*ry*s[0] };
    std::cout<<"stride in : "<< strides_in[0]  << "," << strides_in[1]  << "," << strides_in[2]  <<std::endl;
    std::cout<<"stride out: "<< strides_out[0] << "," << strides_out[1] << "," << strides_out[2] <<std::endl;

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                // image and filter size
                vec3i img_size( az, ay, ax );
                vec3i flt_size( bz, by, bx);
                // out_size[0] = (out_size[0]<=0 ? 1:out_size[0]);
                // out_size[1] = (out_size[1]<=0 ? 1:out_size[1]);
                // out_size[2] = (out_size[2]<=0 ? 1:out_size[2]);

                std::cout<<"image  size: "<< img_size[0] << "," << img_size[1] << "," << img_size[2] <<std::endl;
                std::cout<<"filter size: "<< flt_size[0] << "," << flt_size[1] << "," << flt_size[2] <<std::endl;

                const T* in_ptr  = &(a[xs][ys][zs]);
                T* out_ptr = &(r[xs][ys][zs]);

                int status = vsldConvExec(conv_plans.get(img_size, flt_size),
                                          in_ptr, strides_in,
                                          b.data(), NULL,
                                          out_ptr, strides_out);

            }
    std::cout<<"convolution complete!" << std::endl;
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
