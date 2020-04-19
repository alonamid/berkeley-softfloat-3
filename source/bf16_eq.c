#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

bool bf16_eq( bfloat16_t a, bfloat16_t b )
{
    union ui16_bf16 uA;
    uint_fast16_t uiA;
    union ui16_bf16 uB;
    uint_fast16_t uiB;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    if ( isNaNBF16UI( uiA ) || isNaNBF16UI( uiB ) ) {
        if (
            softfloat_isSigNaNBF16UI( uiA ) || softfloat_isSigNaNBF16UI( uiB )
        ) {
            softfloat_raiseFlags( softfloat_flag_invalid );
        }
        return false;
    }
    return (uiA == uiB) || ! (uint16_t) ((uiA | uiB)<<1);

}
