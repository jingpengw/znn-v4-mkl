// #include "network/parallel/network.hpp"
#include "convolution/convolve_sparse.hpp"
#include "convolution/convolve_sparse_mkl.hpp"
#include <zi/time.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    // cube size
    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;
    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    // filter size
    int64_t fx = 3;
    int64_t fy = 3;
    int64_t fz = 3;
    if ( argc >= 7 )
    {
        fx = atoi(argv[4]);
        fy = atoi(argv[5]);
        fz = atoi(argv[6]);
    }

    // sparseness size
    int64_t sx = 1;
    int64_t sy = 1;
    int64_t sz = 1;
    if ( argc >= 10 )
    {
        sx = atoi(argv[7]);
        sy = atoi(argv[8]);
        sz = atoi(argv[9]);
    }

    // run times
    size_t tc = 10;
    if ( argc == 11 )
    {
        tc = atoi(argv[10]);
    }

    auto v = get_cube<real>(vec3i(x,y,z));
    auto f = get_cube<real>(vec3i(fx,fy,fz));

    for ( uint64_t i = 0; i < v->num_elements(); ++i )
        v->data()[i] = i;//(i%100)/100;
    // auto va = *v;
    // for (int z = 0; z<va.shape()[2]; z++)
    //     for (int y = 0; y<va.shape()[1]; y++)
    //         for (int x = 0; x<va.shape()[0]; x++)
    //         {
    //             std::cout<<"x,y,z,value: "<<x<<", "<<y<<", "<<", "<<z<<", "<<va[x][y][z]<<std::endl;
    //         }

    for ( uint64_t i = 0; i < f->num_elements(); ++i )
        f->data()[i] = (i%100)/10;

    // sparseness
    vec3i s(sx,sy,sz);
    cube_p<real> r1 = convolve_sparse(v,f,s);
    cube_p<real> r2 = convolve_sparse_mkl(v,f,s);

    zi::wall_timer wt;
    wt.reset();

    for ( size_t i = 0; i < tc; ++i )
    {
        r1  = convolve_sparse(v,f,s);
    }
    std::cout << "naive elapsed: " << wt.elapsed<double>() << std::endl;

    wt.reset();
    for ( size_t i = 0; i < tc; ++i )
    {
        r2  = convolve_sparse_mkl(v,f,s);
    }
    std::cout << "mkl   elapsed: " << wt.elapsed<double>() << std::endl;

    std::cout<< "pairwise comparison to detect difference..."<<std::endl;
    for ( uint64_t i = 0; i < r1->num_elements(); ++i )
    {
        if( r1->data()[i] != r2->data()[i] )
            std::cout<< "the result is different: "<< r1->data()[i] << "!=" << r2->data()[i]<<std::endl;
        // else
        //     std::cout<< "the value is the same: "<< r1->data()[i] << "==" << r2->data()[i]<<std::endl;
    }
    std::cout << "Sum compare (naive VS mkl): " << sum(*r1) << "==" << sum(*r2) << " ?" <<std::endl;

}
