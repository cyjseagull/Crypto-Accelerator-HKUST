#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "RapidSV/gsv_wrapper.h"

int char2int(char c) {
    if ('0' <= c && c <= '9')
        return c - '0';
    else if ('a' <= c && c <= 'f')
        return c - 'a' + 10;
    else if ('A' <= c && c <= 'F')
        return c - 'A' + 10;
    else {
        printf("Invalid char: '%c'\n", c);
        exit(1);
    }
}

void hex2bn(uint32_t *x, const char *hex_string, int cnt) {
    int index = 0, length = 0, value;

    for (index = 0; index < cnt; index++) x[index] = 0;

    while (hex_string[length] != 0) length++;

    for (index = 0; index < length; index++) {
        value = char2int(hex_string[length - index - 1]);
        x[index / 8] += value << index % 8 * 4;
    }
}

int main(int argc, char **argv) {
    GSV_init();

    sig_t *sig;
    int *results;

    int count = 1048576;

    sig = (sig_t *)malloc(sizeof(sig_t) * count);
    results = (int *)malloc(sizeof(int) * count);

    for (int i = 0; i < count; i++) {
        hex2bn(sig[i].r._limbs, "23B20B796AAAFEAAA3F1592CB9B4A93D5A8D279843E1C57980E64E0ABC5F5B05", GSV_BITS / 32);
        hex2bn(sig[i].s._limbs, "E11F5909F947D5BE08C84A22CE9F7C338F7CF4A5B941B9268025495D7D433071", GSV_BITS / 32);
        hex2bn(sig[i].e._limbs, "10D51CB90C0C0522E94875A2BEA7AB72299EBE7192E64EFE0573B1C77110E5C9", GSV_BITS / 32);
        hex2bn(sig[i].key_x._limbs, "D5548C7825CBB56150A3506CD57464AF8A1AE0519DFAF3C58221DC810CAF28DD", GSV_BITS / 32);
        hex2bn(sig[i].key_y._limbs, "921073768FE3D59CE54E79A49445CF73FED23086537027264D168946D479533E", GSV_BITS / 32);
    }

    GSV_verify(count, sig, results);

    for (int i = 0; i < count; i++) {
        if (results[i] != 0) {
            printf("Signature #%d does not match public key.\n", i);
        }
    }

    GSV_close();

    return 0;
}
