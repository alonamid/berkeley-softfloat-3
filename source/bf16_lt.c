#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "softfloat.h"

bool bf16_lt( bfloat16_t a, bfloat16_t b )
{
    union ui16_bf16 uA;
    uint_fast16_t uiA;
    union ui16_bf16 uB;
    uint_fast16_t uiB;
    bool signA, signB;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    if ( isNaNBF16UI( uiA ) || isNaNBF16UI( uiB ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        return false;
    }
    signA = signBF16UI( uiA );
    signB = signBF16UI( uiB );
    return
        (signA != signB) ? signA && ((uint16_t) ((uiA | uiB)<<1) != 0)
            : (uiA != uiB) && (signA ^ (uiA < uiB));

}
