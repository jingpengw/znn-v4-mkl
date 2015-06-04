#pragma once

#include "../types.hpp"
#include "../cube/cube_operators.hpp"
#include "convolve_constant.hpp"

#include <zi/utility/singleton.hpp>
#include <mkl_vsl.h>

namespace znn { namespace v4 {

class single_size_conv_plan
{
private:
    std::mutex                      m1       ;
    std::mutex                      m2       ;
    MKL_INT                         ashape[3];
    MKL_INT                         bshape[3];
    MKL_INT                         rshape[3];

public:
    single_size_conv_plan( vec9i const & s )
    {
        ashape[0] = s[0];
        ashape[1] = s[1];
        ashape[2] = s[2];

        bshape[0] = s[3];
        bshape[1] = s[4];
        bshape[2] = s[5];

        rshape[0] = s[6];
        rshape[1] = s[7];
        rshape[2] = s[8];
    }

    VSLConvTaskPtr get()
    {
        guard g(m1);
        int status;
        VSLConvTaskPtr task;
        const int start[3]={bshape[0]-1,bshape[1]-1,bshape[2]-1};

#ifdef ZNN_USE_FLOATS
        status = vslsConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 ashape, bshape, rshape);
#else
        status = vsldConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 ashape, bshape, rshape);
#endif
        status = vslConvSetStart(task, start);
        return task;
    }

    VSLConvTaskPtr get_inv( vec3i const & s )
    {
        guard g(m1);
        int status;
        VSLConvTaskPtr task;
        const int start[3]={bshape[0]-1,bshape[1]-1,bshape[2]-1};

#ifdef ZNN_USE_FLOATS
        status = vslsConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 ashape, bshape, rshape);
#else
        status = vsldConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 ashape, bshape, rshape);
#endif
        status = vslConvSetStart(task, start);
        return task;
    }
};

class conv_plans_impl
{
private:
    std::mutex                               m  ;
    std::map<vec9i, single_size_conv_plan*> pool;

    single_size_conv_plan* get_pool(vec9i const & shape)
    {
        typedef single_size_conv_plan* single_size_conv_plan_ptr;
        guard g(m);
        single_size_conv_plan_ptr& plan_ptr = pool[shape];
        if ( plan_ptr ) return plan_ptr;
        plan_ptr = new single_size_conv_plan(shape);
        return plan_ptr;
    }

    single_size_conv_plan* get_pool(vec3i const & a, vec3i const & b, vec3i const & r)
    {
        vec9i shape;
        shape[0] = a[0];
        shape[1] = a[1];
        shape[2] = a[2];
        shape[3] = b[0];
        shape[4] = b[1];
        shape[5] = b[2];
        shape[6] = r[0];
        shape[7] = r[1];
        shape[8] = r[2];
        return get_pool(shape);
    }

public:
    VSLConvTaskPtr get(vec3i const & a, vec3i const & b, vec3i const & r)
    {
        return get_pool(a,b,r)->get();
    }

    VSLConvTaskPtr get(vec3i const & a, vec3i const & b)
    {
        vec3i r(a[0]-b[0]+1, a[1]-b[1]+1, a[2]-b[2]+1);
        return get_pool(a,b,r)->get();
    }

    VSLConvTaskPtr get_inv(vec3i const & a, vec3i const & b)
    {
        vec3i r(a[0]-b[0]+1, a[1]-b[1]+1, a[2]-b[2]+1);
        return get_pool(a,b,r)->get();
    }

};

namespace {
ZNN_THREAD_LOCAL conv_plans_impl& conv_plans = zi::singleton<conv_plans_impl>::instance();
} // anonymous namespace

template< typename T >
inline cube_p<T> convolve_mkl( cube<T> const & a,
                           cube<T> const & b)
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_copy(a);
        *r *= b[0][0][0];
        return r;
    }

    cube_p<T> rp = get_cube<T>(size(a) + vec3i::one - size(b));

#ifdef ZNN_USE_FLOATS
    int status = vslsConvExec(conv_plans.get(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              rp->data(), NULL);
#else
    int status = vsldConvExec(conv_plans.get(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              rp->data(), NULL);
#endif
    return rp;
}


template< typename T >
inline cube_p<T> convolve_mkl( ccube_p<T> const & a,
                           ccube_p<T> const & b)
{
    return convolve(*a, *b);
}

template< typename T >
inline void convolve_mkl( cube<T> const & a,
                      cube<T> const & b,
                      cube<T> & r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant(a,b.data()[0],r);
        return;
    }

    cube_p<T> rp = convolve_mkl( a, b );
    r = *rp;
}

template< typename T >
inline void convolve_add_mkl( cube<T> const & a,
                          cube<T> const & b,
                          cube<T> & r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant_add_mkl(a,b.data()[0],r);
        return;
    }

    auto radd = convolve_mkl(a,b);
    r += *radd;
}


template< typename T >
inline cube_p<T> convolve_flipped_mkl( cube<T> const & a,
                                   cube<T> const & b)
{
    flip(const_cast<cube<T>&>(b));
    return convolve_mkl(a,b);
}

template< typename T >
inline cube_p<T> convolve_flipped_mkl( ccube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_flipped_mkl(*a, *b);
}


template< typename T >
inline void convolve_flipped_add_mkl( cube<T> const & a,
                                  cube<T> const & b,
                                  cube<T> & r) noexcept
{
    if ( size(a) == size(b) )
    {
        ZI_ASSERT(r.num_elements()==1);
        r.data()[0] += convolve_constant_flipped_mkl(a,b);
        return;
    }

    auto radd = convolve_flipped_mkl(a,b);
    r += *radd;
}


template< typename T >
inline cube_p<T> convolve_inverse_mkl( cube<T> const & a,
                                   cube<T> const & b)
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_copy(a);
        *r *= b[0][0][0];
        return r;
    }

    cube_p<T> rp = get_cube<T>(size(a) + size(b) - vec3i::one);

#ifdef ZNN_USE_FLOATS
    int status = vslsConvExec(conv_plans.get_inv(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              rp->data(), NULL);
#else
    int status = vsldConvExec(conv_plans.get_inv(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              rp->data(), NULL);
#endif

    return rp;
}

template< typename T >
inline cube_p<T> convolve_inverse_mkl( ccube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_inverse_mkl(*a, *b);
}


template< typename T >
inline void convolve_inverse_add_mkl( cube<T> const & a,
                                  cube<T> const & b,
                                  cube<T> & r) noexcept
{
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse_add_mkl(a,b.data()[0],r);
        return;
    }

    auto radd = convolve_inverse_mkl(a,b);
    r += *radd;
}



}} // namespace znn::v4
