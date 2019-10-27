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
        if to_typ == 'i64':
            return 'return vec_ctsl({in0}, 0);'.format(**fmtspec)
        if to_typ == 'i32':
            return 'return vec_cts({in0}, 0);'.format(**fmtspec)
        if to_typ == 'u64':
            return 'return vec_ctul({in0}, 0);'.format(**fmtspec)
        if to_typ == 'u32':
            return 'return vec_ctu({in0}, 0);'.format(**fmtspec)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def downcvt1(simd_ext, from_typ, to_typ):
    if simd_ext == 'vsx':
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == 2 * to_nbits
        if to_typ == 'f32':
            return 'return vec_float2({in0}, {in1});'.format(simd_ext)
        if to_typ[0] != 'f' and from_type[0] != 'f':
            return 'return vec_pack({in0}, {In1});'.format(simd_ext)
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
                  return r;'''.\
               format(from_typ=from_typ, to_typ=to_typ, **fmtspec)
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
        if to_typ[0] != 'f' and from_type[0] != 'f':
            r = '''nsimd_vsx_v{to_typ}x2 r;
                   r.v0 = vec_unpacklo({in0};
                   r.v1 = vec_unpackhi({in0};
                   return r;'''.format(**fmtspec)
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
      'suf': suf(from_typ),
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
        'abs': op1('abs', simd_ext, from_typ),
        'add': op2('add', simd_ext, from_typ),
        'addv': addv1(simd_ext, from_typ),
        'all': op1('all_ne', simd_ext, from_typ),
        'andb': op2('and', simd_ext, from_typ),
        'andl': op2('and', simd_ext, from_typ),
        'andnotb': op2('andc', simd_ext, from_typ),
        'andnotl': op2('andc', simd_ext, from_typ),
        'any': op1('any_ne', simd_ext, from_typ),
        'ceil': op1('ceil', simd_ext, from_typ),
        'div': op2('div', simd_ext, from_typ),
        'eq': op2('cmpeq', simd_ext, from_typ),
        'floor': op1('floor', simd_ext, from_typ),
        'fma': op3('madd', simd_ext, from_typ),
        'fms': op3('msub', simd_ext, from_typ),
        'fnma': op3('nmadd', simd_ext, from_typ),
        'fnms': op3('nmsub', simd_ext, from_typ),
        'ge': op2('cmpge', simd_ext, from_typ),
        'gt': op2('cmpgt', simd_ext, from_typ),
        'if_else1': if_else3(simd_ext, from_typ),
        'le': op2('cmple', simd_ext, from_typ),
        'len': len0(simd_ext, from_typ),
        'lt': op2('cmplt', simd_ext, from_typ),
        'max': op2('max', simd_ext, from_typ),
        'min': op2('min', simd_ext, from_typ),
        'mul': op2('mul', simd_ext, from_typ),
        'ne': op2('cmpne', simd_ext, from_typ),
        'neg': op1('neg', simd_ext, from_typ),
        'notb': not1(simd_ext, from_typ),
        'notl': not1(simd_ext, from_typ),
        'orb': op2('or', simd_ext, from_typ),
        'orl': op2('or', simd_ext, from_typ),
        'rec': rec1(simd_ext, from_typ),
        'rec11': op1('re', simd_ext, from_typ),
        'round': op1('round', simd_ext, from_typ),
        'rsqrt11': op1('rsqrte', simd_ext, from_typ),
        'set1': set1(simd_ext, from_typ),
        'shl': op2('sl', simd_ext, from_typ),
        'shr': op2('sr', simd_ext, from_typ), # logical shift
        'sqrt': op1('sqrt', simd_ext, from_typ),
        'sub': op2('sub', simd_ext, from_typ),
        'trunc': op1('trunc', simd_ext, from_typ),
        'xorb': op2('xor', simd_ext, from_typ),
        'xorl': op2('xor', simd_ext, from_typ),
        'nbtrue': op1('popcnt', simd_ext, from_typ),
        'reverse': op1('reve', simd_ext, from_typ),
        'reinterpret': reinterpret1(simd_ext, from_typ, to_typ),
        'reinterpretl': reinterpret1(simd_ext, from_typ, to_typ),
        'cvt': cvt1(simd_ext, from_typ, to_typ),
        'upcvt': upcvt1(simd_ext, from_typ, to_typ),
        'downcvt': downcvt1(simd_ext, from_typ, to_typ),
        'loada': load(simd_ext, from_typ),
        'loadu': load(simd_ext, from_typ),
        'loadla': load(simd_ext, from_typ),
        'loadlu': load(simd_ext, from_typ),
        'storea': store(simd_ext, from_typ),
        'storeu': store(simd_ext, from_typ),
        'storela': store(simd_ext, from_typ),
        'storelu': store(simd_ext, from_typ),

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
    return impls[func]
