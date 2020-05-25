#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "softfloat.h"

bfloat16_t bf16_sub( bfloat16_t a, bfloat16_t b )
{
    union ui16_bf16 uA;
    uint_fast16_t uiA;
    union ui16_bf16 uB;
    uint_fast16_t uiB;
#if ! defined INLINE_LEVEL || (INLINE_LEVEL < 1)
    bfloat16_t (*magsFuncPtr)( uint_fast16_t, uint_fast16_t );
#endif

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
#if defined INLINE_LEVEL && (1 <= INLINE_LEVEL)
    if ( signBF16UI( uiA ^ uiB ) ) {
        return softfloat_addMagsBF16( uiA, uiB );
    } else {
        return softfloat_subMagsBF16( uiA, uiB );
    }
#else
    magsFuncPtr =
        signBF16UI( uiA ^ uiB ) ? softfloat_addMagsBF16 : softfloat_subMagsBF16;
    return (*magsFuncPtr)( uiA, uiB );
#endif

}