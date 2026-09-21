#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;

fn f32x4_sum(v: v128) -> f32 {
    let a = f32x4_extract_lane::<0>(v);
    let b = f32x4_extract_lane::<1>(v);
    let c = f32x4_extract_lane::<2>(v);
    let d = f32x4_extract_lane::<3>(v);
    a + b + c + d
}

#[no_mangle]
pub unsafe extern "C" fn gemv_f32(m: i32, n: i32, w: i32, x: i32, y: i32) {
    let m = m as usize;
    let n = n as usize;
    let w = w as *const f32;
    let x = x as *const f32;
    let y = y as *mut f32;
    for i in 0..m {
        let row = w.add(i * n);
        let mut acc = f32x4_splat(0.0);
        let mut k = 0usize;
        while k + 16 <= n {
            for _ in 0..4 {
                let wv = v128_load(row.add(k) as *const v128);
                let xv = v128_load(x.add(k) as *const v128);
                acc = f32x4_add(acc, f32x4_mul(wv, xv));
                k += 4;
            }
        }
        let mut s = f32x4_sum(acc);
        while k < n {
            s += *row.add(k) * *x.add(k);
            k += 1;
        }
        *y.add(i) = s;
    }
}

#[no_mangle]
pub unsafe extern "C" fn gemv_q8(m: i32, n: i32, q: i32, scale: i32, x: i32, y: i32) {
    let m = m as usize;
    let n = n as usize;
    let q = q as *const i8;
    let scale = scale as *const f32;
    let x = x as *const f32;
    let y = y as *mut f32;
    for i in 0..m {
        let row = q.add(i * n);
        let mut s = 0.0f32;
        for k in 0..n {
            s += (*row.add(k) as f32) * *x.add(k);
        }
        *y.add(i) = s * *scale.add(i);
    }
}

#[no_mangle]
pub unsafe extern "C" fn gemv_q4(
    m: i32,
    n: i32,
    packed: i32,
    scale: i32,
    group: i32,
    x: i32,
    y: i32,
) {
    let m = m as usize;
    let n = n as usize;
    let group = group as usize;
    let packed = packed as *const u8;
    let scale = scale as *const f32;
    let x = x as *const f32;
    let y = y as *mut f32;
    let ng = n / group;
    let packed_row = n / 2;
    for i in 0..m {
        let mut acc = 0.0f32;
        let row = packed.add(i * packed_row);
        let so = i * ng;
        for g in 0..ng {
            let sc = *scale.add(so + g);
            let base = row.add(g * (group / 2));
            let mut s = 0.0f32;
            let mut k = 0usize;
            while k < group {
                let b = *base.add(k / 2);
                s += (((b & 15) as i32) - 8) as f32 * *x.add(g * group + k);
                s += (((b >> 4) as i32) - 8) as f32 * *x.add(g * group + k + 1);
                k += 2;
            }
            acc += s * sc;
        }
        *y.add(i) = acc;
    }
}

#[no_mangle]
pub unsafe extern "C" fn rms_norm(n: i32, x: i32, w: i32, out: i32, tokens: i32) {
    let n = n as usize;
    let tokens = tokens as usize;
    let x = x as *const f32;
    let w = w as *const f32;
    let out = out as *mut f32;
    let eps = 1e-6f32;
    for t in 0..tokens {
        let xo = x.add(t * n);
        let oo = out.add(t * n);
        let mut ss = 0.0f32;
        for i in 0..n {
            let v = *xo.add(i);
            ss += v * v;
        }
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            *oo.add(i) = *xo.add(i) * inv * *w.add(i);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn silu_mul(n: i32, a: i32, b: i32) {
    let n = n as usize;
    let a = a as *mut f32;
    let b = b as *const f32;
    for i in 0..n {
        let v = *a.add(i);
        *a.add(i) = (v / (1.0 + (-v).exp())) * *b.add(i);
    }
}

#[no_mangle]
pub extern "C" fn kernel_version() -> i32 {
    1
}
