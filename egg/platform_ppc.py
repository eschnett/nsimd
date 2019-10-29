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
#   - VMX -> 128 bits registers
#   - VSX -> 128 bits registers
#
# VSX also supports 128-bit integer and floating point types (i128, u128, f128),
# but these are not yet made available here.
#
# Relevant documentation:
# - IBM: https://www.ibm.com/support/knowledgecenter/SSGH2K_13.1.2/com.ibm.xlc131.aix.doc/compiler_ref/vec_intrin_cpp.html
# - GCC: https://gcc.gnu.org/onlinedocs/gcc-9.2.0/gcc/PowerPC-AltiVec_002fVSX-Built-in-Functions.html#PowerPC-AltiVec_002fVSX-Built-in-Functions
# - PowerISA ABI (e.g. via Wikipedia; TODO: add link)

import common

vmx = ['vmx', 'vsx']
vsx = ['vsx']

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def get_simd_exts():
    return ['vmx', 'vsx']

def emulate_fp16(simd_ext):
    if simd_ext in vmx:
        return True
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_type(simd_ext, typ):
    if simd_ext in vmx:
        if typ == 'f16':
            return 'struct {{ __vector float v[2]; }}'.format(typ=typ)
        return '__vector {typ}'.format(typ=typ)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_logical_type(simd_ext, typ):
    if simd_ext in vmx:
        # The vector syntax requires changes to the compiler parser
        nbits = int(typ[1:])
        if typ == 'f16':
            return 'struct {{ __vector __bool int v[2]; }}'.\
                   format(nbits=nbits)
        if nbits == 64:
            return '__vector __bool long long';
        if nbits == 32:
            return '__vector __bool int';
        if nbits == 16:
            return '__vector __bool short';
        if nbits == 8:
            return '__vector __bool char';
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_nb_registers(simd_ext):
    if simd_ext in vmx:
        return '32'
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def has_compatible_SoA_types(simd_ext):
    if simd_ext in vmx:
        return False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_additional_include(func, platform, simd_ext):
    ret = '#include <string.h>\n'
    if simd_ext in vmx:
        ret += '#include <nsimd/cpu/cpu/{}.h>\n'.format(func)
    if simd_ext in vsx:
        ret += '#include <nsimd/ppc/vmx/{}.h>\n'.format(func)
    return ret

# -----------------------------------------------------------------------------
## Special functions

def count1(simd_ext, typ):
    if simd_ext in vmx:
        nbits = int(typ[1:])
        if nbits == 64:
            return 'return !!{in0}[0] + !!{in0}[1];'.format(**fmtspec)
        if nbits == 32:
            return 'return (!!{in0}[0] + !!{in0}[1]) + '\
                          '(!!{in0}[2] + !!{in0}[3]);'''.format(**fmtspec)
        if nbits == 16:
            return 'return ((!!{in0}[0] + !!{in0}[1]) + '\
                           '(!!{in0}[2] + !!{in0}[3])) + '\
                          '((!!{in0}[4] + !!{in0}[5]) + '\
                           '(!!{in0}[6] + !!{in0}[7]));'.format(**fmtspec)
        if nbits == 8:
            return 'return (((!!{in0}[0] + !!{in0}[1]) + '\
                            '(!!{in0}[2] + !!{in0}[3])) + '\
                           '((!!{in0}[4] + !!{in0}[5]) + '\
                            '(!!{in0}[6] + !!{in0}[7]))) + '\
                          '(((!!{in0}[8] + !!{in0}[9]) + '\
                           '(!!{in0}[10] + !!{in0}[11])) + '\
                          '((!!{in0}[12] + !!{in0}[13]) + '\
                           '(!!{in0}[14] + !!{in0}[15])));'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def cvt1(simd_ext, from_typ, to_typ):
    if simd_ext in vmx:
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == to_nbits
        if to_typ[0] == from_typ[0]:
            # no conversion
            return 'return {in0};'.format(**fmtspec)
        if to_typ[0] != 'f' and from_typ[0] != 'f':
            # convert between signed and unsigned integer types: reinterpret
            return '''{styp} x = {in0};
                      {dtyp} r;
                      memcpy(&r, &x, sizeof(r));
                      return r;'''.format(**fmtspec)
        # everything below converts between int and float
        if to_typ == 'f64':
            # return 'return vec_ctd({in0}, 0);'.format(**fmtspec)
            return 'return vec_double({in0});'.format(**fmtspec)
        if to_typ == 'f32':
            # return 'return vec_ctf({in0}, 0);'.format(**fmtspec)
            return 'return vec_float({in0});'.format(**fmtspec)
        if to_typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        if to_typ == 'i64':
            # return 'return vec_ctsl({in0}, 0);'.format(**fmtspec)
            return common.NOT_IMPLEMENTED # TODO
        if to_typ == 'i32':
            return 'return vec_cts({in0}, 0);'.format(**fmtspec)
        if to_typ == 'i16':
            return common.NOT_IMPLEMENTED # TODO
        if to_typ == 'u64':
            # return 'return vec_ctul({in0}, 0);'.format(**fmtspec)
            return common.NOT_IMPLEMENTED # TODO
        if to_typ == 'u32':
            return 'return vec_ctu({in0}, 0);'.format(**fmtspec)
        if to_typ == 'u16':
            return common.NOT_IMPLEMENTED # TODO
        print("cvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def downcvt1(simd_ext, from_typ, to_typ):
    if simd_ext in vmx:
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == 2 * to_nbits
        if to_typ == 'f32':
            return 'return vec_float2({in0}, {in1});'.format(**fmtspec)
        if to_typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        if to_typ[0] != 'f' and from_typ[0] != 'f':
            from_ityp = 'i' + from_typ[1:]
            to_ityp = 'i' + to_typ[1:]
            return '''nsimd_{simd_ext}_v{from_typ} x0, x1;
                      nsimd_{simd_ext}_v{from_ityp} xi0, xi1;
                      memcpy(&xi0, &x0, sizeof(xi0));
                      memcpy(&xi1, &x1, sizeof(xi1));
                      nsimd_{simd_ext}_v{to_ityp} ri = vec_pack(xi0, xi1);
                      nsimd_{simd_ext}_v{to_typ} r;
                      memcpy(&r, &ri, sizeof(r));
                      return r;'''.\
                   format(from_ityp=from_ityp, to_ityp=to_ityp, **fmtspec)
        if to_typ[0] != 'f':
            return common.NOT_IMPLEMENTED # TODO
        print("downcvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def if_else3(simd_ext, typ):
    # x ? y : z
    if simd_ext in vmx:
        if typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        return 'return vec_sel({in2}, {in1}, {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def len0(simd_ext, typ):
    if simd_ext in vmx:
        nbits = int(typ[1:])
        nelts = 128 // nbits
        return 'return {nelts};'.format(nelts=nelts)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def madd3(op, simd_ext, typ):
    if simd_ext in vmx:
        if typ[0] == 'f':
            if typ == 'f16':
                return '''nsimd_{simd_ext}_v{typ} r;
                          r.v[0] = vec_{op}({in0}.v[0], {in1}.v[0], {in2}.v[0]);
                          r.v[1] = vec_{op}({in0}.v[1], {in1}.v[1], {in2}.v[1]);
                          return r;'''.format(op=op, **fmtspec)
            return 'return vec_{op}({in0}, {in1}, {in2});'.\
                format(op=op, **fmtspec)
        if op == 'madd':
            return 'return vec_add(vec_mul({in0}, {in1}), {in2});'.\
                format(op=op, **fmtspec)
        if op == 'msub':
            return 'return vec_sub(vec_mul({in0}, {in1}), {in2});'.\
                format(op=op, **fmtspec)
        if op == 'nmadd':
            if typ[0] == 'u':
                return common.NOT_IMPLEMENTED # undefined
            return 'return vec_neg(vec_add(vec_mul({in0}, {in1}), {in2}));'.\
                format(op=op, **fmtspec)
        if op == 'nmsub':
            return 'return vec_sub({in2}, vec_mul({in0}, {in1}));'.\
                format(op=op, **fmtspec)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def not1(simd_ext, typ, logical=''):
    if simd_ext in vmx:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{logical}{typ} r;
                      r.v[0] = vec_nor({in0}.v[0], {in0}.v[0]);
                      r.v[1] = vec_nor({in0}.v[1], {in0}.v[1]);
                      return r;'''.format(logical=logical, **fmtspec)
        return 'return vec_nor({in0}, {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def rec1(simd_ext, typ):
    if simd_ext in vmx:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{typ} r;
                      r.v[0] = vec_div(vec_splats((f32)1), {in0}.v[0]);
                      r.v[1] = vec_div(vec_splats((f32)1), {in0}.v[1]);
                      return r;'''.format(**fmtspec)
        return 'return vec_div(vec_splats(({typ})1), {in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def reinterpret1(simd_ext, from_typ, to_typ):
    if simd_ext in vmx:
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == to_nbits
        if to_typ == from_typ:
            return 'return {in0};'.format(**fmtspec)
        if from_typ == 'f16' or to_typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        return '''{styp} x = {in0};
                  {dtyp} r;
                  memcpy(&r, &x, sizeof(r));
                  return r;'''.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def reinterpretl1(simd_ext, from_typ, to_typ):
    if simd_ext in vmx:
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert from_nbits == to_nbits
        if from_typ == 'f16' or to_typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        return '''{ltyp} x = {in0};
                  {dltyp} r;
                  memcpy(&r, &x, sizeof(r));
                  return r;'''.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def round1(op, simd_ext, typ):
    if simd_ext in vmx:
        if typ[0] != 'f':
            # rounding for integers is trivial
            return 'return {in0};'.format(**fmtspec)
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{typ} r;
                      r.v[0] = vec_{op}({in0}.v[0]);
                      r.v[1] = vec_{op}({in0}.v[1]);
                      return r;'''.format(op=op, **fmtspec)
        return 'return vec_{op}({in0});'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def set1(simd_ext, typ):
    if simd_ext in vmx:
        if typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        return 'return vec_splats({in0});'.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def shift2(op, simd_ext, typ):
    if simd_ext in vmx:
        if typ[0] == 'f':
            return common.NOT_IMPLEMENTED # TODO
        utyp = 'u' + typ[1:];
        return 'return vec_{op}({in0}, vec_splats(({utyp}){in1}));'.\
               format(op=op, utyp=utyp, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def upcvt1(simd_ext, from_typ, to_typ):
    if simd_ext in vmx:
        from_nbits = int(from_typ[1:])
        to_nbits = int(to_typ[1:])
        assert 2 * from_nbits == to_nbits
        if to_typ == 'f64':
            return '''nsimd_{simd_ext}_v{to_typ}x2 r;
                      r.v0 = vec_doublel({in0});
                      r.v1 = vec_doubleh({in0});
                      return r;'''.format(**fmtspec)
        if to_typ == 'f32':
            return common.NOT_IMPLEMENTED # TODO
        if to_typ == 'f16':
            return common.NOT_IMPLEMENTED # TODO
        if to_typ[0] != 'f' and from_typ[0] != 'f':
            from_ityp = 'i' + from_typ[1:]
            to_ityp = 'i' + to_typ[1:]
            return '''nsimd_{simd_ext}_v{from_typ} x;
                      nsimd_{simd_ext}_v{from_ityp} xi;
                      memcpy(&xi, &x, sizeof(xi));
                      nsimd_{simd_ext}_v{to_ityp}x2 ri;
                      ri.v0 = vec_unpackl(xi);
                      ri.v1 = vec_unpackh(xi);
                      nsimd_{simd_ext}_v{to_typ}x2 r;
                      memcpy(&r, &ri, sizeof(r));
                      return r;'''.\
                   format(from_ityp=from_ityp, to_ityp=to_ityp, **fmtspec)
        if to_typ in ['i64', 'u64', 'i32', 'u32', 'i16', 'u16']:
            return common.NOT_IMPLEMENTED # TODO
        print("upcvt1.ft", from_typ, to_typ)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# true = vec_eqv(u,u);
# false = vec_xor(u,u);

# -----------------------------------------------------------------------------
## Standard functions

def op1(op, simd_ext, typ, logical=''):
    if simd_ext in vmx:
        if op == 'neg' and typ[0] == 'u':
            nbits = typ[1:]
            return '''nsimd_{simd_ext}_vu{nbits} x = {in0};
                      nsimd_{simd_ext}_vi{nbits} xi;
                      memcpy(&xi, &x, sizeof(xi));
                      nsimd_{simd_ext}_vi{nbits} ri = vec_neg(xi);
                      nsimd_{simd_ext}_vu{nbits} r;
                      memcpy(&r, &ri, sizeof(r));
                      return r;'''.format(nbits=nbits, **fmtspec)
        if op == 'abs' and typ[0] == 'u':
            return 'return {in0};'.format(**fmtspec)
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{logical}{typ} r;
                      r.v[0] = vec_{op}({in0}.v[0]);
                      r.v[1] = vec_{op}({in0}.v[1]);
                      return r;'''.format(op=op, logical=logical, **fmtspec)
        return 'return vec_{op}({in0});'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def op2(op, simd_ext, typ, logical=''):
    if simd_ext in vmx:
        nbits = int(typ[1:])
        nelts = 128 // nbits
        if op == 'div' and typ[0] != 'f' and nbits != 64:
            stmts = ['{styp} x = {in0};',
                     '{styp} y = {in1};',
                     '{dtyp} r;']
            stmts.extend(['r[{}] = x[{}] / y[{}];'.format(i, i, i)
                          for i in range(0, nelts)])
            stmts.append('return r;')
            return '\n'.join(stmts).format(**fmtspec)
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{logical}{typ} r;
                      r.v[0] = vec_{op}({in0}.v[0], {in1}.v[0]);
                      r.v[1] = vec_{op}({in0}.v[1], {in1}.v[1]);
                      return r;'''.format(op=op, logical=logical, **fmtspec)
        return 'return vec_{op}({in0}, {in1});'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}";'.format(simd_ext))

def op3(op, simd_ext, typ):
    if simd_ext in vmx:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{typ} r;
                      r.v[0] = vec_{op}({in0}.v[0], {in1}.v[0], {in2}.v[0]);
                      r.v[1] = vec_{op}({in0}.v[1], {in1}.v[1], {in2}.v[1]);
                      return r;'''.format(op=op, **fmtspec)
        return 'return vec_{op}({in0}, {in1}, {in2});'.format(op=op, **fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def red1(op, simd_ext, typ):
    if simd_ext in vmx:
        nbits = int(typ[1:])
        if op in ['addv', 'nbtrue']:
            nelts = 128 // nbits
            if typ == 'f16':
                nelts2 = nelts // 2
                args = ['{{in0}}.v[{}][{}]'.format(i // nelts2, i % nelts2)
                        for i in range(0, nelts)]
            else:
                args = ['{{in0}}[{}]'.format(i) for i in range(0, nelts)]
            if op == 'nbtrue':
                args = ['(int){}'.format(arg) for arg in args]
            while len(args) > 1:
                args = ['({} + {})'.format(args[i], args[i+1])
                        for i in range(0, len(args), 2)]
            r = args[0]
            if op == 'addv' and typ == 'f16':
                r = 'nsimd_f32_to_f16({})'.format(r)
            if op == 'nbtrue':
                r = '-{}'.format(r)
            return 'return {};'.format(r).format(**fmtspec)
        if op in ['all', 'any']:
            cop = '&' if op == 'all' else '|'
            if typ == 'f16':
                return '''nsimd_{simd_ext}_vi32 zero = vec_splats((i32)0);
                          nsimd_{simd_ext}_vli32 zerol;
                          memcpy(&zerol, &zero, sizeof(zerol));
                          return -(vec_{op}_ne({in0}.v[0], zerol) {cop}
                                   vec_{op}_ne({in0}.v[1], zerol));'''.\
                       format(op=op, cop=cop, **fmtspec)
            lstyp = typ if typ[0] != 'f' else 'i' + typ[1:]
            return 'return -vec_{op}_ne({in0}, vec_splats(({lstyp})0));'.\
                format(op=op, lstyp=lstyp, **fmtspec)
        assert False
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# -----------------------------------------------------------------------------
## Load/store

def load(simd_ext, from_typ):
    if simd_ext in vmx:
        if from_typ == 'f16':
            nbits = int(from_typ[1:])
            nelts = 128 // nbits
            nelts2 = nelts // 2
            stmts = ['nsimd_{simd_ext}_v{from_typ} r;']
            stmts.extend(['r.v[{}][{}] = '\
                              'nsimd_u16_to_f32(((const u16*){{in0}})[{}]);'.\
                          format(i // nelts2, i % nelts2, i)
                          for i in range(0, nelts)])
            stmts.append('return r;')
            return '\n'.join(stmts).format(**fmtspec)
        return 'return *(const nsimd_{simd_ext}_v{from_typ}*){in0};'.\
               format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def store(simd_ext, from_typ):
    if simd_ext in vmx:
        if from_typ == 'f16':
            nbits = int(from_typ[1:])
            nelts = 128 // nbits
            nelts2 = nelts // 2
            stmts = ['((u16*){{in0}})[{}] = '\
                         'nsimd_f32_to_u16({{in1}}.v[{}][{}]);'.\
                     format(i, i // nelts2, i % nelts2)
                     for i in range(0, nelts)]
            return '\n'.join(stmts).format(**fmtspec)
        return '*(nsimd_{simd_ext}_v{from_typ}*){in0} = {in1};'.\
               format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def loadl(simd_ext, from_typ):
    if simd_ext in vmx:
        if from_typ == 'f16':
            nbits = int(from_typ[1:])
            nelts = 128 // nbits
            nelts2 = nelts // 2
            return '''nsimd_{simd_ext}_vu16 x =
                          *(const nsimd_{simd_ext}_vu16*){in0};
                      nsimd_{simd_ext}_vi16 xi;
                      memcpy(&xi, &x, sizeof(xi));
                      nsimd_{simd_ext}_vl{from_typ} r;
                      r.v[0] = vec_cmpne(vec_unpackl(xi), vec_splats((i32)0));
                      r.v[1] = vec_cmpne(vec_unpackh(xi), vec_splats((i32)0));
                      return r;'''.format(**fmtspec)
        return 'return vec_cmpne(*(const nsimd_{simd_ext}_v{from_typ}*){in0}, '\
                                'vec_splats(({from_typ})0));'.format(**fmtspec);
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def storel(simd_ext, from_typ):
    if simd_ext in vmx:
        if from_typ == 'f16':
            return '''nsimd_{simd_ext}_vu16 oneu =
                          vec_splats(nsimd_f32_to_u16(1));
                      nsimd_{simd_ext}_vlu16 x =
                          vec_pack({in1}.v[0], {in1}.v[1]);
                      *(nsimd_{simd_ext}_vu16*){in0} = vec_and(oneu, x);'''.\
                   format(**fmtspec)
        return '''nsimd_{simd_ext}_v{from_typ} one =
                      vec_splats(({from_typ})1);
                  *(nsimd_{simd_ext}_v{from_typ}*){in0} =
                      vec_and(one, {in1});'''.format(**fmtspec)
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

# -----------------------------------------------------------------------------
## get_impl function

def get_impl(func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
      'simd_ext': simd_ext,
      'typ': from_typ,
      'ltyp': get_logical_type(simd_ext, from_typ),
      'styp': get_type(simd_ext, from_typ),
      'dtyp': get_type(simd_ext, to_typ),
      'dltyp': get_logical_type(simd_ext, to_typ),
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
        'addv': lambda: red1('addv', simd_ext, from_typ), # sum
        'all': lambda: red1('all', simd_ext, from_typ),
        'andb': lambda: op2('and', simd_ext, from_typ),
        'andl': lambda: op2('and', simd_ext, from_typ, 'l'),
        'andnotb': lambda: op2('andc', simd_ext, from_typ),
        'andnotl': lambda: op2('andc', simd_ext, from_typ, 'l'),
        'any': lambda: red1('any', simd_ext, from_typ),
        'ceil': lambda: round1('ceil', simd_ext, from_typ),
        'div': lambda: op2('div', simd_ext, from_typ),
        'eq': lambda: op2('cmpeq', simd_ext, from_typ, 'l'),
        'floor': lambda: round1('floor', simd_ext, from_typ),
        'fma': lambda: madd3('madd', simd_ext, from_typ),
        'fms': lambda: madd3('msub', simd_ext, from_typ),
        'fnma': lambda: madd3('nmsub', simd_ext, from_typ),
        'fnms': lambda: madd3('nmadd', simd_ext, from_typ),
        'ge': lambda: op2('cmpge', simd_ext, from_typ, 'l'),
        'gt': lambda: op2('cmpgt', simd_ext, from_typ, 'l'),
        'if_else1': lambda: if_else3(simd_ext, from_typ),
        'le': lambda: op2('cmple', simd_ext, from_typ, 'l'),
        'len': lambda: len0(simd_ext, from_typ),
        'lt': lambda: op2('cmplt', simd_ext, from_typ, 'l'),
        'max': lambda: op2('max', simd_ext, from_typ),
        'min': lambda: op2('min', simd_ext, from_typ),
        'mul': lambda: op2('mul', simd_ext, from_typ),
        'ne': lambda: op2('cmpne', simd_ext, from_typ, 'l'),
        'neg': lambda: op1('neg', simd_ext, from_typ),
        'notb': lambda: not1(simd_ext, from_typ),
        'notl': lambda: not1(simd_ext, from_typ, 'l'),
        'orb': lambda: op2('or', simd_ext, from_typ),
        'orl': lambda: op2('or', simd_ext, from_typ, 'l'),
        'rec': lambda: rec1(simd_ext, from_typ),
        'rec11': lambda: op1('re', simd_ext, from_typ),
        'round_to_even': lambda: round1('round', simd_ext, from_typ),
        'rsqrt11': lambda: op1('rsqrte', simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'shl': lambda: shift2('sl', simd_ext, from_typ),
        'shr': lambda: shift2('sr', simd_ext, from_typ), # logical shift
        'sqrt': lambda: op1('sqrt', simd_ext, from_typ),
        'sub': lambda: op2('sub', simd_ext, from_typ),
        'trunc': lambda: round1('trunc', simd_ext, from_typ),
        'xorb': lambda: op2('xor', simd_ext, from_typ),
        'xorl': lambda: op2('xor', simd_ext, from_typ, 'l'),
        'nbtrue': lambda: red1('nbtrue', simd_ext, from_typ),
        'reverse': lambda: op1('reve', simd_ext, from_typ),
        'reinterpret': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        'reinterpretl': lambda: reinterpretl1(simd_ext, from_typ, to_typ),
        'cvt': lambda: cvt1(simd_ext, from_typ, to_typ),
        'upcvt': lambda: upcvt1(simd_ext, from_typ, to_typ),
        'downcvt': lambda: downcvt1(simd_ext, from_typ, to_typ),
        'loada': lambda: load(simd_ext, from_typ),
        'loadu': lambda: load(simd_ext, from_typ),
        'loadla': lambda: loadl(simd_ext, from_typ),
        'loadlu': lambda: loadl(simd_ext, from_typ),
        'storea': lambda: store(simd_ext, from_typ),
        'storeu': lambda: store(simd_ext, from_typ),
        'storela': lambda: storel(simd_ext, from_typ),
        'storelu': lambda: storel(simd_ext, from_typ),

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
