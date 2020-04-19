
#=============================================================================
#
# This Makefile template is part of the SoftFloat IEEE Floating-Point
# Arithmetic Package, Release 3e, by John R. Hauser.
#
# Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
# University of California.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions, and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions, and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the University nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
# DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#=============================================================================

# Edit lines marked with `==>'.  See "SoftFloat-source.html".
XLEN ?= 64

SOURCE_DIR ?= ../../source
SPECIALIZE_TYPE ?= RISCV

SOFTFLOAT_OPTS ?= \
  -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 \
  -DSOFTFLOAT_FAST_DIV64TO32

DELETE = rm -f
C_INCLUDES = -I. -I$(SOURCE_DIR)/$(SPECIALIZE_TYPE) -I$(SOURCE_DIR)/include

CC_LINUX_PRESENT := $(shell command -v riscv$(XLEN)-unknown-linux-gnu-gcc 2> /dev/null)

ifdef CC_LINUX_PRESENT
    CC_LINUX := riscv$(XLEN)-unknown-linux-gnu-gcc
else
    CC_LINUX := riscv$(XLEN)-linux-gnu-gcc
endif

COMPILE_C = $(CC_LINUX) -c $(SOFTFLOAT_OPTS) $(C_INCLUDES) -O2 -o $@

MAKELIB = ar crs $@

OBJ = .o
LIB = .a

OTHER_HEADERS =

.PHONY: all
all: libsoftfloat$(LIB)

OBJS_PRIMITIVES = \
  s_shortShiftRightJam64$(OBJ) \
  s_shiftRightJam16$(OBJ) \
  s_shiftRightJam32$(OBJ) \
  s_shiftRightJam64$(OBJ) \
  s_countLeadingZeros8$(OBJ) \
  s_countLeadingZeros16$(OBJ) \
  s_countLeadingZeros32$(OBJ) \
  s_countLeadingZeros64$(OBJ) \

OBJS_SPECIALIZE = \
  s_commonNaNToF32UI$(OBJ) \
  s_f32UIToCommonNaN$(OBJ) \
  softfloat_raiseFlags$(OBJ) \
  s_propagateNaNF32UI$(OBJ) \
  s_propagateNaNBF16UI$(OBJ) \

OBJS_OTHERS = \
  s_roundToUI32$(OBJ) \
  s_roundToI32$(OBJ) \
  s_normSubnormalBF16Sig$(OBJ) \
  s_roundPackToBF16$(OBJ) \
  s_normRoundPackToBF16$(OBJ) \
  s_subMagsBF16$(OBJ) \
  s_addMagsBF16$(OBJ) \
  s_mulAddBF16$(OBJ) \
  softfloat_state$(OBJ) \
  bf16_to_f32$(OBJ) \
  bf16_add$(OBJ) \
  bf16_eq$(OBJ) \
  bf16_eq_signaling$(OBJ) \
  bf16_le$(OBJ) \
  bf16_le_quiet$(OBJ) \
  bf16_lt$(OBJ) \
  bf16_lt_quiet$(OBJ) \
  bf16_mul$(OBJ) \
  bf16_mulAdd$(OBJ) \
  f32_to_bf16$(OBJ) \
  i32_to_bf16$(OBJ) \

OBJS_ALL = $(OBJS_PRIMITIVES) $(OBJS_SPECIALIZE) $(OBJS_OTHERS)

$(OBJS_ALL): \
  $(OTHER_HEADERS) platform.h $(SOURCE_DIR)/include/primitiveTypes.h \
  $(SOURCE_DIR)/include/primitives.h
$(OBJS_SPECIALIZE) $(OBJS_OTHERS): \
  $(SOURCE_DIR)/include/softfloat_types.h $(SOURCE_DIR)/include/internals.h \
  $(SOURCE_DIR)/$(SPECIALIZE_TYPE)/specialize.h \
  $(SOURCE_DIR)/include/softfloat.h

$(OBJS_PRIMITIVES) $(OBJS_OTHERS): %$(OBJ): $(SOURCE_DIR)/%.c
	$(COMPILE_C) $(SOURCE_DIR)/$*.c

$(OBJS_SPECIALIZE): %$(OBJ): $(SOURCE_DIR)/$(SPECIALIZE_TYPE)/%.c
	$(COMPILE_C) $(SOURCE_DIR)/$(SPECIALIZE_TYPE)/$*.c

libsoftfloat$(LIB): $(OBJS_ALL)
	$(DELETE) $@
	$(MAKELIB) $^

.PHONY: clean
clean:
	$(DELETE) $(OBJS_ALL) libsoftfloat$(LIB)

