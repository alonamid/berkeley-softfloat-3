#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "softfloat.h"

bfloat16_t bf16_mulAdd( bfloat16_t a, bfloat16_t b, bfloat16_t c )
{
    union ui16_bf16 uA;
    uint_fast16_t uiA;
    union ui16_bf16 uB;
    uint_fast16_t uiB;
    union ui16_bf16 uC;
    uint_fast16_t uiC;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    uC.f = c;
    uiC = uC.ui;
    return softfloat_mulAddBF16( uiA, uiB, uiC, 0 );

}