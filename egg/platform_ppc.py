# Copyright (c) 2019 Agenium Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file gives the implementation of platform POWER, i.e. Power SIMD.
# This file implements the following architectures:
#   - Altivec -> not yet implemented
#   - VSX     -> 128 bits registers

# VSX also supports 128-bit integer and floating point type (i128, u128, f128),
# but these are not yet made available here.

import common

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def get_simd_exts():
    return ['vsx']

def emulate_fp16(simd_ext):
    if simd_ext == 'vsx':
        return False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_type(simd_ext, typ):
    if simd_ext == 'vsx':
        # The vector syntax requires changes to the compiler parser
        return '__vector {typ}'.format(typ=typ)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_logical_type(simd_ext, typ):
    if simd_ext == 'vsx':
        # The vector syntax requires changes to the compiler parser
        nbits = int(typ[1:])
        if nbits == 64:
            return '__vector __bool long long'
        if nbits == 32:
            return '__vector __bool int'
        if nbits == 16:
            return '__vector __bool short'
        if nbits == 8:
            return '__vector __bool char'
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_nb_registers(simd_ext):
    if simd_ext == 'vsx':
        return '32'
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def has_compatible_SoA_types(simd_ext):
    if simd_ext == 'vsx':
        return False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_additional_include(func, platform, simd_ext):
    ret = ''
    if simd_ext == 'vsx':
        ret += '''#include <string.h>
                  #include <nsimd/cpu/cpu/{}.h>
                  '''.format(func)
    return ret

# -----------------------------------------------------------------------------
## Special functions

def addv1(simd_ext, typ):
    # sum
    if simd_ext == 'vsx':
        nbits = int(typ[1:])
        if nbits == 64:
            return 'return {in0}[0] + {in0}[1];'.format(**fmtspec)
        if nbits == 32:
            return 'return ({in0}[0] + {in0}[1]) + ({in0}[2] + {in0}[3]);'''.format(**fmtspec)
        if nbits == 16:
            return 'return (({in0}[0] + {in0}[1]) + ({in0}[2] + {in0}[3])) + (({in0}[4] + {in0}[5]) + ({in0}[6] + {in0}[7]));'.format(**fmtspec)
        if nbits == 8:
            return 'return ((({in0}[0] + {in0}[1]) + ({in0}[2] + {in0}[3])) + (({in0}[4] + {in0}[5]) + ({in0}[6] + {in0}[7]))) + ((({in0}[8] + {in0}[9]) + ({in0}[10] + {in0}[11])) + (({in0}[12] + {in0}[13]) + ({in0}[14] + {in0}[15])));'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def cvt1(simd_ext, from_typ, to_typ):
    if simd_ext == 'vsx':
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == to_nbits
        if (to_typ[0] == 'f') == (from_typ[0] == 'f'):
            # from float to float, or from [u]int to [u]int
            return 'return {in0};'.format(**fmtspec)
        # everything below converts between int and float
        if to_typ == 'f64':
            # return 'return vec_ctd({in0}, 0);'.format(**fmtspec)
            return 'return vec_double({in0});'.format(**fmtspec)
        if to_typ == 'f32':
            # return 'return vec_ctf({in0}, 0);'.format(**fmtspec)
            return 'return vec_float({in0});'.format(**fmtspec)
        if to_typ == 'f16':
            return 'abort();'   # TODO
        if to_typ == 'i64':
            return 'return vec_ctsl({in0}, 0);'.format(**fmtspec)
        if to_typ == 'i32':
            return 'return vec_cts({in0}, 0);'.format(**fmtspec)
        if to_typ == 'i16':
            return 'abort();'   # TODO
        if to_typ == 'u64':
            return 'return vec_ctul({in0}, 0);'.format(**fmtspec)
        if to_typ == 'u32':
            return 'return vec_ctu({in0}, 0);'.format(**fmtspec)
        if to_typ == 'u16':
            return 'abort();'   # TODO
        print("cvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def downcvt1(simd_ext, from_typ, to_typ):
    if simd_ext == 'vsx':
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == 2 * to_nbits
        if to_typ == 'f32':
            return 'return vec_float2({in0}, {in1});'.format(**fmtspec)
        if to_typ == 'f16':
            return 'abort();'   # TODO
        if to_typ[0] != 'f' and from_typ[0] != 'f':
            return 'return vec_pack({in0}, {in1});'.format(**fmtspec)
        if to_typ[0] != 'f':
            return 'abort();'   # TODO
        print("downcvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def if_else3(simd_ext, typ):
    # x ? y : z
    if simd_ext == 'vsx':
        return 'return vec_sel({in2}, {in1}, {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def len0(simd_ext, typ):
    if simd_ext == 'vsx':
        nbits = int(typ[1:])
        nelts = 128 // nbits
        return 'return {nelts};'.format(nelts=nelts)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def not1(simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_nor({in0}, {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def rec1(simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_div(vec_splats((typ)1), {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def reinterpret1(simd_ext, from_typ, to_typ):
    if simd_ext == 'vsx':
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == to_nbits
        return '''__vector {from_typ} x = {in0};
                  __vector {to_typ} r;
                  memcpy(&r, &x, sizeof {to_typ});
                  return r;'''.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def set1(simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_splats({in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def upcvt1(simd_ext, from_typ, to_typ):
    if simd_ext == 'vsx':
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert 2 * from_nbits == to_nbits
        if to_typ == 'f64':
            return '''nsimd_vsx_v{to_typ}x2 r;
                      r.v0 = vec_doublel({in0});
                      r.v1 = vec_doubleh({in0});
                      return r;'''.format(**fmtspec)
        if to_typ == 'f32':
            return 'abort();'   # TODO
        if to_typ == 'f16':
            return 'abort();'   # TODO
        if to_typ[0] != 'f' and from_typ[0] != 'f':
            return '''nsimd_vsx_v{to_typ}x2 r;
                      r.v0 = vec_unpacklo({in0};
                      r.v1 = vec_unpackhi({in0};
                      return r;'''.format(**fmtspec)
        if to_typ in ['i64', 'u64', 'i32', 'u32', 'i16', 'u16']:
            return 'abort();'   # TODO
        print("upcvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# true = vec_eqv(u,u);
# false = vec_xor(u,u);

# -----------------------------------------------------------------------------
## Standard functions

def op1(op, simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_{op}({in0})'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def op2(op, simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_{op}({in0}, {in1})'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def op3(op, simd_ext, typ):
    if simd_ext == 'vsx':
        return 'return vec_{op}({in0}, {in1}, {in2});'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# -----------------------------------------------------------------------------
## Load/store

def load(simd_ext, from_typ):
    if simd_ext == 'vsx':
        return 'return *(const vector {typ}*){in0};'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def store(simd_ext, from_typ):
    if simd_ext == 'vsx':
        return '*(vector {typ}*){in0} = {in1};'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# -----------------------------------------------------------------------------
## get_impl function

def get_impl(func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
      'simd_ext': simd_ext,
      'typ': from_typ,
      'styp': get_type(simd_ext, from_typ),
      'from_typ': from_typ,
      'to_typ': to_typ,
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'in3': common.in3,
      'in4': common.in4,
      'in5': common.in5,
      'typnbits': from_typ[1:],
      'svtrue': 'svptrue_b{}()'.format(from_typ[1:])
    }

    impls = {
        'abs': lambda: op1('abs', simd_ext, from_typ),
        'add': lambda: op2('add', simd_ext, from_typ),
        'addv': lambda: addv1(simd_ext, from_typ),
        'all': lambda: op1('all_ne', simd_ext, from_typ),
        'andb': lambda: op2('and', simd_ext, from_typ),
        'andl': lambda: op2('and', simd_ext, from_typ),
        'andnotb': lambda: op2('andc', simd_ext, from_typ),
        'andnotl': lambda: op2('andc', simd_ext, from_typ),
        'any': lambda: op1('any_ne', simd_ext, from_typ),
        'ceil': lambda: op1('ceil', simd_ext, from_typ),
        'div': lambda: op2('div', simd_ext, from_typ),
        'eq': lambda: op2('cmpeq', simd_ext, from_typ),
        'floor': lambda: op1('floor', simd_ext, from_typ),
        'fma': lambda: op3('madd', simd_ext, from_typ),
        'fms': lambda: op3('msub', simd_ext, from_typ),
        'fnma': lambda: op3('nmadd', simd_ext, from_typ),
        'fnms': lambda: op3('nmsub', simd_ext, from_typ),
        'ge': lambda: op2('cmpge', simd_ext, from_typ),
        'gt': lambda: op2('cmpgt', simd_ext, from_typ),
        'if_else1': lambda: if_else3(simd_ext, from_typ),
        'le': lambda: op2('cmple', simd_ext, from_typ),
        'len': lambda: len0(simd_ext, from_typ),
        'lt': lambda: op2('cmplt', simd_ext, from_typ),
        'max': lambda: op2('max', simd_ext, from_typ),
        'min': lambda: op2('min', simd_ext, from_typ),
        'mul': lambda: op2('mul', simd_ext, from_typ),
        'ne': lambda: op2('cmpne', simd_ext, from_typ),
        'neg': lambda: op1('neg', simd_ext, from_typ),
        'notb': lambda: not1(simd_ext, from_typ),
        'notl': lambda: not1(simd_ext, from_typ),
        'orb': lambda: op2('or', simd_ext, from_typ),
        'orl': lambda: op2('or', simd_ext, from_typ),
        'rec': lambda: rec1(simd_ext, from_typ),
        'rec11': lambda: op1('re', simd_ext, from_typ),
        'round': lambda: op1('round', simd_ext, from_typ),
        'rsqrt11': lambda: op1('rsqrte', simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'shl': lambda: op2('sl', simd_ext, from_typ),
        'shr': lambda: op2('sr', simd_ext, from_typ), # logical shift
        'sqrt': lambda: op1('sqrt', simd_ext, from_typ),
        'sub': lambda: op2('sub', simd_ext, from_typ),
        'trunc': lambda: op1('trunc', simd_ext, from_typ),
        'xorb': lambda: op2('xor', simd_ext, from_typ),
        'xorl': lambda: op2('xor', simd_ext, from_typ),
        'nbtrue': lambda: op1('popcnt', simd_ext, from_typ),
        'reverse': lambda: op1('reve', simd_ext, from_typ),
        'reinterpret': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        'reinterpretl': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        'cvt': lambda: cvt1(simd_ext, from_typ, to_typ),
        'upcvt': lambda: upcvt1(simd_ext, from_typ, to_typ),
        'downcvt': lambda: downcvt1(simd_ext, from_typ, to_typ),
        'loada': lambda: load(simd_ext, from_typ),
        'loadu': lambda: load(simd_ext, from_typ),
        'loadla': lambda: load(simd_ext, from_typ),
        'loadlu': lambda: load(simd_ext, from_typ),
        'storea': lambda: store(simd_ext, from_typ),
        'storeu': lambda: store(simd_ext, from_typ),
        'storela': lambda: store(simd_ext, from_typ),
        'storelu': lambda: store(simd_ext, from_typ),

        # TODO
        # 'load2a': load1234(simd_ext, from_typ, 2),
        # 'load3a': load1234(simd_ext, from_typ, 3),
        # 'load4a': load1234(simd_ext, from_typ, 4),
        # 'load2u': load1234(simd_ext, from_typ, 2),
        # 'load3u': load1234(simd_ext, from_typ, 3),
        # 'load4u': load1234(simd_ext, from_typ, 4),
        # 'store2a': store1234(simd_ext, from_typ, 2),
        # 'store3a': store1234(simd_ext, from_typ, 3),
        # 'store4a': store1234(simd_ext, from_typ, 4),
        # 'store2u': store1234(simd_ext, from_typ, 2),
        # 'store3u': store1234(simd_ext, from_typ, 3),
        # 'store4u': store1234(simd_ext, from_typ, 4),
        # 'to_logical': to_logical1(simd_ext, from_typ),
        # 'to_mask': to_mask1(simd_ext, from_typ),
        # 'ziplo': zip_unzip_half("zip1", simd_ext, from_typ),
        # 'ziphi': zip_unzip_half("zip2", simd_ext, from_typ),
        # 'unziplo': zip_unzip_half("uzp1", simd_ext, from_typ),
        # 'unziphi': zip_unzip_half("uzp2", simd_ext, from_typ)
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if from_typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if func not in impls:
        return common.NOT_IMPLEMENTED
    return impls[func]()
