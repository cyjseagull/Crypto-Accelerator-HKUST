#include <cuda.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cgbn/cgbn.h"
#include "gsv_wrapper.h"
#include "support.h"

#define GSV_SM2  // enable optimization for SM2 (a=-3)

#define GSV_TABLE_SIZE 512  // size of precomputed table

#define GSV_MAX_INS 1048576  // maximum number of instants in a single kernel

__constant__ cgbn_mem_t<GSV_BITS> d_mul_table[GSV_TABLE_SIZE];
#ifdef GSV_KNOWN_PKEY
__constant__ cgbn_mem_t<GSV_BITS> d_mul_table2[GSV_TABLE_SIZE];
#endif

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)
// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
template <uint32_t tpi>
class gsv_params_t {
   public:
    // parameters used by the CGBN context
    static const uint32_t TPB = 0;            // get TPB from blockDim.x
    static const uint32_t MAX_ROTATION = 4;   // good default value
    static const uint32_t SHM_LIMIT = 0;      // no shared mem available
    static const bool CONSTANT_TIME = false;  // constant time implementations aren't available yet

    // parameters used locally in the application
    static const uint32_t TPI = tpi;  // threads per instance
};

typedef struct {
    cgbn_mem_t<GSV_BITS> r;  // sig->r
    cgbn_mem_t<GSV_BITS> s;  // sig->s
    cgbn_mem_t<GSV_BITS> e;  // digest
#ifndef GSV_KNOWN_PKEY
    cgbn_mem_t<GSV_BITS> key_x;  // public key
    cgbn_mem_t<GSV_BITS> key_y;  // public key
#endif
} instance_t;

typedef struct {
    cgbn_mem_t<GSV_BITS> order;  // group order
    cgbn_mem_t<GSV_BITS> field;  // prime p
    cgbn_mem_t<GSV_BITS> g_a;
} ec_t;

template <class params>
class gsv_t {
   public:
    typedef cgbn_context_t<params::TPI> context_t;
    typedef cgbn_env_t<context_t, GSV_BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;
    typedef typename env_t::cgbn_local_t bn_local_t;
    typedef typename env_t::cgbn_wide_t bn_wide_t;

    context_t _context;
    env_t _env;
    int32_t _instance;

    __device__ __forceinline__ gsv_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance)
        : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}

    __device__ __forceinline__ void mod(bn_t &r, const bn_t &m) {
        while (_env.compare(r, m) >= 0) {
            _env.sub(r, r, m);
        }
    }

    // fast modular addition: r = (a + b) mod m
    // both a and b should be non-negative and less than m
    __device__ __forceinline__ void mod_add(bn_t &r, const bn_t &a, const bn_t &b, const bn_t &m) {
#ifdef GSV_256BIT
        if (_env.add(r, a, b) || _env.compare(r, m) >= 0) {
            _env.sub(r, r, m);
        }
#else
        _env.add(r, a, b);
        if (_env.compare(r, m) >= 0) {
            _env.sub(r, r, m);
        }
#endif
    }

    // r = (a - b) mod m
    __device__ __forceinline__ void mod_sub(bn_t &r, const bn_t &a, const bn_t &b, const bn_t &m) {
        if (_env.sub(r, a, b)) {  // a < b
            _env.add(r, r, m);
        }
    }

    // r = (a * 2) mod m
    __device__ __forceinline__ void mod_lshift1(bn_t &r, const bn_t &a, const bn_t &m) {
#ifdef GSV_256BIT
        uint32_t z = _env.clz(a);
        _env.shift_left(r, a, 1);
        if (z == 0 || _env.compare(r, m) >= 0) {
            _env.sub(r, r, m);
        }
#else
        _env.shift_left(r, a, 1);
        if (_env.compare(r, m) >= 0) {
            _env.sub(r, r, m);
        }
#endif
    }

    // Intel IPP's faster point doubling
    // Complexity: 6S, 4M, 2A, 3D, 3L, 1R
    // SM2:        4S, 4M, 2A, 4D, 3L, 1R
    __device__ __forceinline__ void point_dbl_ipp(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y,
                                                  const bn_t &a_z, const bn_t &field, const bn_t &g_a, const uint32_t np0) {
        if (_env.equals_ui32(a_z, 0)) {
            _env.set_ui32(r_z, 0);
            return;
        }

        bn_t u, m, s, t;

        mod_lshift1(s, a_y, field);         // s = 2 * a_y
        _env.mont_sqr(u, a_z, field, np0);  // u = a_z^2

        _env.mont_sqr(m, s, field, np0);         // m = 4 * a_y^2
        _env.mont_mul(r_z, s, a_z, field, np0);  // r_z = 2 * a_y * a_z

        _env.mont_sqr(r_y, m, field, np0);  // r_y = 16 * a_y^4

        _env.mont_mul(s, m, a_x, field, np0);  // s = 4 * a_x * a_y^2

#ifdef GSV_256BIT
        if (_env.ctz(r_y) == 0 && _env.add(r_y, r_y, field)) {
            _env.shift_right(r_y, r_y, 1);
            _env.bitwise_mask_ior(r_y, r_y, -1);
        } else {
            _env.shift_right(r_y, r_y, 1);  // r_y = 8 * a_y^4
        }
#else
        if (_env.ctz(r_y) == 0) {
            _env.add(r_y, r_y, field);
        }
        _env.shift_right(r_y, r_y, 1);      // r_y = 8 * a_y^4
#endif

#ifdef GSV_SM2
        mod_add(m, a_x, u, field);           // m = a_x + u
        mod_sub(u, a_x, u, field);           // u = a_x - u
        _env.mont_mul(m, m, u, field, np0);  // m = (a_x + u) * (a_x - u) = a_x^2 - a_z^4
        mod_lshift1(t, m, field);            // t = 2 * (a_x^2 - a_z^4)
        mod_add(m, m, t, field);             // m = 3 * (a_x^2 - a_z^4)
#else
        _env.mont_sqr(m, a_x, field, np0);  // m = a_x ^ 2
        mod_lshift1(t, m, field);           // t = 2 * a_x^2
        mod_add(m, m, t, field);            // m = 3 * a_x^2

        _env.mont_sqr(u, u, field, np0);       // u = a_z^4
        _env.mont_mul(u, u, g_a, field, np0);  // u = g_a * a_z^4
        mod_add(m, m, u, field);               // m = 3 * a_x^2 + g_a * a_z^4
#endif

        mod_lshift1(u, s, field);           // u = 8 * a_x * a_y^2
        _env.mont_sqr(r_x, m, field, np0);  // r_x = m^2
        mod_sub(r_x, r_x, u, field);        // r_x = m^2 - u

        mod_sub(s, s, r_x, field);           // s = 4 * a_x * a_y^2 - r_x
        _env.mont_mul(s, s, m, field, np0);  // s = (4 * a_x * a_y^2 - r_x) * m
        mod_sub(r_y, s, r_y, field);         // r_y = (4 * a_x * a_y^2 - r_x) * m - 8 * a_y^4
    }

    // Intel IPP's faster point addition
    // Complexity: 4S, 12M, 0A, 6D, 1L
    __device__ __forceinline__ void point_add_ipp(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y,
                                                  const bn_t &a_z, const bn_t &b_x, const bn_t &b_y, const bn_t &b_z,
                                                  const bn_t &field, const bn_t &g_a, const uint32_t np0) {
        if (_env.compare(a_x, b_x) == 0 && _env.compare(a_y, b_y) == 0 && _env.compare(a_z, b_z) == 0) {
            // if (threadIdx.x == 0) printf("DOUBLE\n");
            point_dbl_ipp(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
            return;
        }
        if (_env.equals_ui32(a_z, 0)) {
            _env.set(r_x, b_x);
            _env.set(r_y, b_y);
            _env.set(r_z, b_z);
            return;
        }
        if (_env.equals_ui32(b_z, 0)) {
            _env.set(r_x, a_x);
            _env.set(r_y, a_y);
            _env.set(r_z, a_z);
            return;
        }

        bn_t u1, u2, s1, s2, h, r;

        _env.mont_mul(s1, a_y, b_z, field, np0);  // s1 = a_y * b_z
        _env.mont_sqr(u1, b_z, field, np0);       // u1 = b_z^2

        _env.mont_mul(s2, b_y, a_z, field, np0);  // s2 = b_y * a_z
        _env.mont_sqr(u2, a_z, field, np0);       // u2 = a_z^2

        _env.mont_mul(s1, s1, u1, field, np0);  // s1 = a_y * b_z^3
        _env.mont_mul(s2, s2, u2, field, np0);  // s2 = b_y * a_z^3

        _env.mont_mul(u1, a_x, u1, field, np0);  // u1 = a_x * b_z^2
        _env.mont_mul(u2, b_x, u2, field, np0);  // u2 = b_x * a_z^2

        mod_sub(r, s2, s1, field);  // r = s2 - s1
        mod_sub(h, u2, u1, field);  // h = u2 - u1

        if (_env.equals_ui32(h, 0)) {
            if (_env.equals_ui32(r, 0)) {
                // if (threadIdx.x == 0) printf("EQUAL\n");
                point_dbl_ipp(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
                return;
            } else {
                _env.set_ui32(r_z, 0);
                return;
            }
        }

        _env.mont_mul(r_z, a_z, b_z, field, np0);  // r_z = a_z * b_z
        _env.mont_sqr(u2, h, field, np0);          // u2 = h^2
        _env.mont_mul(r_z, r_z, h, field, np0);    // r_z = a_z * b_z * h
        _env.mont_sqr(s2, r, field, np0);          // s2 = r^2
        _env.mont_mul(h, h, u2, field, np0);       // h = h^3

        _env.mont_mul(u1, u1, u2, field, np0);  // u1 = u1 * h^2
        mod_sub(r_x, s2, h, field);             // r_x = r^2 - h^3
        mod_lshift1(u2, u1, field);             // u2 = 2 * u1 * h^2
        _env.mont_mul(s1, s1, h, field, np0);   // s1 = s1 * h^3
        mod_sub(r_x, r_x, u2, field);           // r_x = r^2 - h^3 - 2 * u1 * h^2

        mod_sub(r_y, u1, r_x, field);            // r_y = u1 * h^2 - r_x
        _env.mont_mul(r_y, r_y, r, field, np0);  // r_y = r * (u1 * h^2 - r_x)
        mod_sub(r_y, r_y, s1, field);            // r_y = r * (u1 * h^2 - r_x) - s1 * h^3
    }

    // Non-adjacent form (NAF)
    // Expected complexity: n * D + n/3 * A
    __device__ __forceinline__ void point_mult_naf(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &p_x, const bn_t &p_y,
                                                   const bn_t &p_z, const bn_t &d, const bn_t &field, const bn_t &g_a,
                                                   const uint32_t np0) {
        bn_t q_x, q_y, q_z;
        bn_t k, m_y;
        int8_t naf[257];

        _env.set(q_x, p_x);
        _env.set(q_y, p_y);
        _env.set(q_z, p_z);
        _env.set(k, d);
        _env.set_ui32(r_z, 0);

        _env.sub(m_y, field, p_y);  // my = -p_y mod field

        int bits = 0;
        while (_env.compare_ui32(k, 0) > 0) {
            if (_env.ctz(k) == 0) {  // k is odd
                _env.shift_right(k, k, 1);
                if (_env.ctz(k) == 0) {  // k mod 4 = 3
                    naf[bits] = -1;
                    _env.add_ui32(k, k, 1);
                } else {  // k mod 4 = 1;
                    naf[bits] = 1;
                }
            } else {
                _env.shift_right(k, k, 1);
                naf[bits] = 0;
            }
            ++bits;
        }

        for (int i = bits - 1; i >= 0; i--) {
            point_dbl_ipp(r_x, r_y, r_z, r_x, r_y, r_z, field, g_a, np0);
            if (naf[i] == 1) {
                point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, q_y, q_z, field, g_a, np0);
            } else if (naf[i] == -1) {
                point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, m_y, q_z, field, g_a, np0);
            }
        }
    }

    // Fixed-point multiplication, can be applied for the generator point
    // Expected complexity: n/2 * A
    // Table size: 32 KB (512 bn)
    __device__ __forceinline__ void fixed_point_mult(bn_t &r_x, bn_t &r_y, bn_t &r_z, bn_t &k, const bn_t &field,
                                                     const bn_t &g_a, const uint32_t np0) {
        int i = 0;
        bn_t q_x, q_y, one;

        _env.set(one, r_z);
        _env.set_ui32(r_z, 0);

        while (_env.compare_ui32(k, 0) > 0) {
            if (_env.ctz(k) == 0) {  // k_i = 1
                _env.load(q_x, &d_mul_table[i * 2]);
                _env.load(q_y, &d_mul_table[i * 2 + 1]);
                point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, q_y, one, field, g_a, np0);
            }
            _env.shift_right(k, k, 1);
            i++;
        }
    }

#ifdef GSV_KNOWN_PKEY
    __device__ __forceinline__ void fixed_point_mult2(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &d, const bn_t &field,
                                                      const bn_t &g_a, const uint32_t np0) {
        int i = 0;
        bn_t k, q_x, q_y, one;

        _env.set(one, r_z);
        _env.set(k, d);
        _env.set_ui32(r_z, 0);

        while (_env.compare_ui32(k, 0) > 0) {
            if (_env.ctz(k) == 0) {  // k_i = 1
                _env.load(q_x, &d_mul_table2[i * 2]);
                _env.load(q_y, &d_mul_table2[i * 2 + 1]);
                point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, q_y, one, field, g_a, np0);
            }
            _env.shift_right(k, k, 1);
            i++;
        }
    }
#endif

    /*
     * B1: verify whether r' in [1,n-1], verification failed if not
     * B2: verify whether s' in [1,n-1], verification failed if not
     * B3: set M'~=ZA || M'
     * B4: calculate e'=Hv(M'~)
     * B5: calculate t = (r' + s') modn, verification failed if t=0
     * B6: calculate the point (x1', y1')=[s']G + [t]PA
     * B7: calculate R=(e'+x1') modn, verification pass if yes, otherwise failed
     */
#ifndef GSV_KNOWN_PKEY
    __device__ __forceinline__ int32_t sig_verify(const bn_t &r, bn_t &s, const bn_t &e, const bn_t &key_x, const bn_t &key_y,
                                                  const bn_t &order, const bn_t &field, bn_t &g_a)
#else
    __device__ __forceinline__ int32_t sig_verify(const bn_t &r, bn_t &s, const bn_t &e, const bn_t &order, const bn_t &field,
                                                  bn_t &g_a)
#endif
    {
        bn_t t, x1, y1, z1, x2, y2, z2;
        uint32_t np0;

        if (_env.compare_ui32(r, 1) < 0 || _env.compare_ui32(s, 1) < 0 || _env.compare(order, r) <= 0 ||
            _env.compare(order, s) <= 0) {
            return 0;
        }

        mod_add(t, r, s, order);

        if (_env.equals_ui32(t, 0)) {
            return 0;
        }

        _env.set_ui32(z1, 1);
        np0 = _env.bn2mont(z1, z1, field);
        _env.set(z2, z1);

        // mod(g_a, field);  // unnecessary
        _env.bn2mont(g_a, g_a, field);

        // s * generator + t * pkey
        fixed_point_mult(x1, y1, z1, s, field, g_a, np0);

        __syncthreads();  // TODO: temp fix of wrong answer, need to test on different input

#ifndef GSV_KNOWN_PKEY
        _env.set(x2, key_x);
        _env.set(y2, key_y);
        mod(x2, field);
        _env.bn2mont(x2, x2, field);
        mod(y2, field);
        _env.bn2mont(y2, y2, field);
        point_mult_naf(x2, y2, z2, x2, y2, z2, t, field, g_a, np0);
#else
        fixed_point_mult2(x2, y2, z2, t, field, g_a, np0);
#endif

        point_add_ipp(x1, y1, z1, x1, y1, z1, x2, y2, z2, field, g_a, np0);

        // avoid coordinate transformation by converting (r-e) to Jacobian
        mod_sub(t, r, e, order);
        _env.bn2mont(t, t, field);
        _env.mont_sqr(z1, z1, field, np0);
        _env.mont_mul(t, t, z1, field, np0);
        return _env.compare(x1, t);
    }

    static cgbn_mem_t<GSV_BITS> *prepare_table() {
        cgbn_mem_t<GSV_BITS> *mul_table = (cgbn_mem_t<GSV_BITS> *)malloc(sizeof(cgbn_mem_t<GSV_BITS>) * GSV_TABLE_SIZE);

#include "sm2_base_512.table"

        return mul_table;
    }

#ifdef GSV_KNOWN_PKEY
    static cgbn_mem_t<GSV_BITS> *prepare_table2() {
        cgbn_mem_t<GSV_BITS> *mul_table = (cgbn_mem_t<GSV_BITS> *)malloc(sizeof(cgbn_mem_t<GSV_BITS>) * GSV_TABLE_SIZE);

#include "sm2_pkey_512.table"

        return mul_table;
    }
#endif
};

template <class params>
__global__ void kernel_sig_verify(cgbn_error_report_t *report, instance_t *instances, uint32_t instance_count, ec_t ec,
                                  int32_t *results) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
    if (instance >= instance_count) return;

    typedef gsv_t<params> local_gsv_t;

    local_gsv_t gsv(cgbn_report_monitor, report, instance);
#ifndef GSV_KNOWN_PKEY
    typename local_gsv_t::bn_t r, s, e, key_x, key_y, order, field, g_a;
#else
    typename local_gsv_t::bn_t r, s, e, order, field, g_a;
#endif

    cgbn_load(gsv._env, r, &(instances[instance].r));
    cgbn_load(gsv._env, s, &(instances[instance].s));
    cgbn_load(gsv._env, e, &(instances[instance].e));
#ifndef GSV_KNOWN_PKEY
    cgbn_load(gsv._env, key_x, &(instances[instance].key_x));
    cgbn_load(gsv._env, key_y, &(instances[instance].key_y));
#endif

    cgbn_load(gsv._env, order, &(ec.order));
    cgbn_load(gsv._env, field, &(ec.field));
    cgbn_load(gsv._env, g_a, &(ec.g_a));

#ifndef GSV_KNOWN_PKEY
    results[instance] = gsv.sig_verify(r, s, e, key_x, key_y, order, field, g_a);
#else
    results[instance] = gsv.sig_verify(r, s, e, order, field, g_a);
#endif
}

// global variables
ec_t sm2;
instance_t *d_instances;
int32_t *d_results;
cgbn_error_report_t *report;

void GSV_init(int device_id) {
    typedef gsv_params_t<GSV_TPI> params;

    CUDA_CHECK(cudaSetDevice(device_id));

    CUDA_CHECK(cudaMalloc((void **)&d_instances, sizeof(instance_t) * GSV_MAX_INS));
    CUDA_CHECK(cudaMalloc((void **)&d_results, sizeof(int32_t) * GSV_MAX_INS));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    set_words(sm2.order._limbs, "FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", GSV_BITS / 32);
    set_words(sm2.field._limbs, "FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", GSV_BITS / 32);
    set_words(sm2.g_a._limbs, "FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", GSV_BITS / 32);

    cgbn_mem_t<GSV_BITS> *mul_table = gsv_t<params>::prepare_table();
#ifdef GSV_KNOWN_PKEY
    cgbn_mem_t<GSV_BITS> *mul_table2 = gsv_t<params>::prepare_table2();
#endif

    CUDA_CHECK(cudaMemcpyToSymbol(d_mul_table, mul_table, sizeof(cgbn_mem_t<GSV_BITS>) * GSV_TABLE_SIZE));
#ifdef GSV_KNOWN_PKEY
    CUDA_CHECK(cudaMemcpyToSymbol(d_mul_table2, mul_table2, sizeof(cgbn_mem_t<GSV_BITS>) * GSV_TABLE_SIZE));
#endif

    free(mul_table);
#ifdef GSV_KNOWN_PKEY
    free(mul_table2);
#endif
}

void GSV_verify(int count, sig_t *sig, int *results) {
    typedef gsv_params_t<GSV_TPI> params;

    int TPB = (params::TPB == 0) ? 128 : params::TPB;  // default threads per block is 128
    int TPI = params::TPI;
    int IPB = TPB / TPI;  // IPB: instances per block

    CUDA_CHECK(cudaMemcpy(d_instances, sig, sizeof(instance_t) * count, cudaMemcpyHostToDevice));

    kernel_sig_verify<params><<<(count + IPB - 1) / IPB, TPB>>>(report, d_instances, count, sm2, d_results);

    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    CUDA_CHECK(cudaMemcpy(results, d_results, sizeof(int32_t) * count, cudaMemcpyDeviceToHost));
}

void GSV_close() {
    CUDA_CHECK(cudaFree(d_instances));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cgbn_error_report_free(report));
}
