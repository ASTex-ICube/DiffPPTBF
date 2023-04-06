#include "opencl/noise.h"

////////////////////////////////////////////////////////////////////////////////
// MACRO
////////////////////////////////////////////////////////////////////////////////

// PRNG (Pseudo-Random Number Generator) + Noise
#define MAX_NOISE_RAND 1024

// Point Process
// - tiling type
#define PP_tilingType_REGULAR 0
#define PP_tilingType_IRREGULAR 1
#define PP_tilingType_CROSS 2
#define PP_tilingType_BISQUARE 3
#define PP_tilingType_IRREGULARX 4
#define PP_tilingType_IRREGULARY 5
// - max number of neighbors
#define PP_nbMaxNeighbors 18

////////////////////////////////////////////////////////////////////////////////
// PRNG (Pseudo-Random Number Generator)
////////////////////////////////////////////////////////////////////////////////

uint g_PRNG_seed;

//------------------------------------------------------------------------------
// phi
//------------------------------------------------------------------------------
int phi(int x) {
  if (x < 0) {
    x = x + 10 * MAX_NOISE_RAND;
  }
  x = x % MAX_NOISE_RAND;
  return noise[x];
}

//------------------------------------------------------------------------------
// seeding
//------------------------------------------------------------------------------
void seeding(uint x, uint y, uint z) {
  g_PRNG_seed =
      (uint)(phi((int)x + phi((int)y + phi((int)z))) % (int)(1 << 15) +
             (phi(3 * (int)x + phi(4 * (int)y + phi((int)z))) %
              (int)(1 << 15)) *
                 (int)(1 << 15));
}

//------------------------------------------------------------------------------
// next
//------------------------------------------------------------------------------
float next() {
  g_PRNG_seed *= 3039177861u;
  float res = ((float)g_PRNG_seed / (float)4294967296.0f) * 2.0f - 1.0f;
  return res;
}

////////////////////////////////////////////////////////////////////////////////
// Perlin noise
////////////////////////////////////////////////////////////////////////////////
float2 inoiseG(int ix, int iy) {
  int index = (phi(ix) + 3 * phi(iy)) % MAX_NOISE_RAND;
  return G[index];
}

float cnoise2DG(float x, float y) {
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  x -= ix;
  y -= iy;

  float sx = (x * x * (3.0f - 2.0f * x));
  float sy = (y * y * (3.0f - 2.0f * y));

  float2 vy0 = inoiseG(ix, iy);
  float2 vy1 = inoiseG(ix, iy + 1);
  float vx0 =
      mix(dot(vy0, (float2)(x, y)), dot(vy1, (float2)(x, y - 1.0f)), sy);

  vy0 = inoiseG(ix + 1, iy);
  vy1 = inoiseG(ix + 1, iy + 1);
  float vx1 = mix(dot(vy0, (float2)(x - 1.0f, y)),
                  dot(vy1, (float2)(x - 1.0f, y - 1.0f)), sy);

  float rt = mix(vx0, vx1, sx);

  return rt;
}

////////////////////////////////////////////////////////////////////////////////
// POINT PROCESS
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// PP_pave
//------------------------------------------------------------------------------
void PP_pave(
    // position
    float xp, float yp,
    // pavement parameters
    int Nx, float correction, float randx, float randy, float *cx, float *cy,
    float *dx, float *dy) {

  int i, j;
  int nc = 0;
  float x = xp;
  float y = yp;

  int ix = (int)(floor(x));
  float xx = x - (float)(ix);
  int iy = (int)(floor(y));
  float yy = y - (float)(iy);

  for (j = -1; j <= +1; j++) {
    for (i = -1; i <= +1; i++) {
      float rxi, rxs, ryi, rys;
      float ivpx = (float)(ix) + (float)(i);
      float ivpy = (float)(iy) + (float)(j);
      float decalx = (float)((int)(ivpy) % Nx) / (float)(Nx);
      seeding((uint)(ivpx + 5.0f), (uint)(ivpy + 10.0f), 0u);
      rxi = next() * randx * 0.5f;
      seeding(3u, (uint)(ivpy + 10.0f), 0u);
      ryi = next() * randy * 0.5f;
      seeding((uint)(ivpx + 1.0f + 5.0f), (uint)(ivpy + 10.0f), 0u);
      rxs = next() * randx * 0.5f;
      seeding(3u, (uint)(ivpy + 1.0f + 10.0f), 0u);
      rys = next() * randy * 0.5f;

      dx[nc] = 0.5f * (rxs + 1.0f - rxi);
      dy[nc] = 0.5f * (rys + 1.0f - ryi);
      cx[nc] = ivpx + decalx + rxi + dx[nc] - correction;
      cy[nc] = ivpy + ryi + dy[nc];
      nc++;
    }
  }
}

//------------------------------------------------------------------------------
// PP_paved
//------------------------------------------------------------------------------
void PP_paved(float x, float y,
              // pavement parameters
              int Nx, float *cx, float *cy, float *dx, float *dy) {

  int i, j;
  int ix = (int)(floor(x));
  float xx = x - (float)(ix);
  int iy = (int)(floor(y));
  float yy = y - (float)(iy);
  int qx = (int)(xx * (float)(2 * Nx));
  int qy = (int)(yy * (float)(2 * Nx));

  // horizontal
  if ((qx >= qy && qx <= qy + Nx - 1) ||
      (qx >= qy - 2 * Nx && qx <= qy + Nx - 1 - 2 * Nx)) {
    int rx, ry;

    if (qx >= qy && qx <= qy + Nx - 1) {
      rx = qy;
      ry = qy;
    } else {
      rx = qy - 2 * Nx;
      ry = qy;
    }

    for (i = 0; i < 3; i++) {
      cx[3 * i] =
          (float)(ix) +
          ((float)(rx) + (float)(i - 1) + (float)(Nx)*0.5f) / (float)(2 * Nx);
      cy[3 * i] =
          (float)(iy) + ((float)(ry) + (float)(i - 1) + 0.5f) / (float)(2 * Nx);
      dx[3 * i] = ((float)(Nx)*0.5f) / (float)(2 * Nx);
      dy[3 * i] = 0.5f / (float)(2 * Nx);

      cx[3 * i + 1] =
          (float)(ix) + ((float)(rx) + (float)(i - 2) + 0.5f) / (float)(2 * Nx);
      cy[3 * i + 1] =
          (float)(iy) +
          ((float)(ry) + (float)(i - 1) + (float)(Nx)*0.5f) / (float)(2 * Nx);
      dx[3 * i + 1] = 0.5f / (float)(2 * Nx);
      dy[3 * i + 1] = ((float)(Nx)*0.5f) / (float)(2 * Nx);

      cx[3 * i + 2] =
          (float)(ix) +
          ((float)(rx) + (float)(i - 1) + (float)(Nx) + 0.5f) / (float)(2 * Nx);
      cy[3 * i + 2] =
          (float)(iy) +
          ((float)(ry) + (float)(i) - (float)(Nx)*0.5f) / (float)(2 * Nx);
      dx[3 * i + 2] = 0.5f / (float)(2 * Nx);
      dy[3 * i + 2] = ((float)(Nx)*0.5f) / (float)(2 * Nx);
    }
  }
  // vertical
  else {
    int rx, ry;
    if (qy >= qx + 1 && qy <= qx + 1 + Nx - 1) {
      rx = qx;
      ry = qx + 1;
    } else {
      rx = qx;
      ry = qx + 1 - 2 * Nx;
    }
    for (i = 0; i < 3; i++) {
      cx[3 * i] =
          (float)(ix) + ((float)(rx) + (float)(i - 1) + 0.5f) / (float)(2 * Nx);
      cy[3 * i] =
          (float)(iy) +
          ((float)(ry) + (float)(i - 1) + (float)(Nx)*0.5f) / (float)(2 * Nx);
      dx[3 * i] = 0.5f / (float)(2 * Nx);
      dy[3 * i] = ((float)(Nx)*0.5f) / (float)(2 * Nx);

      cx[3 * i + 1] =
          (float)(ix) +
          ((float)(rx) + (float)(i - 1) + (float)(Nx)*0.5f) / (float)(2 * Nx);
      cy[3 * i + 1] =
          (float)(iy) + ((float)(ry) + (float)(i - 2) + 0.5f) / (float)(2 * Nx);
      dx[3 * i + 1] = ((float)(Nx)*0.5f) / (float)(2 * Nx);
      dy[3 * i + 1] = 0.5f / (float)(2 * Nx);

      cx[3 * i + 2] =
          (float)(ix) +
          ((float)(rx) + (float)(i - 1) - (float)(Nx)*0.5f) / (float)(2 * Nx);
      cy[3 * i + 2] = (float)(iy) +
                      ((float)(ry) + (float)(i - 1) + (float)(Nx - 1) + 0.5f) /
                          (float)(2 * Nx);
      dx[3 * i + 2] = ((float)(Nx)*0.5f) / (float)(2 * Nx);
      dy[3 * i + 2] = 0.5f / (float)(2 * Nx);
    }
  }
}

//------------------------------------------------------------------------------
// PP_paveb
//------------------------------------------------------------------------------
void PP_paveb(
    // position
    float x, float y,
    // pavement parameters
    float *cx, float *cy, float *dx, float *dy) {
  int i, j;
  int nc = 0;
  int ii, jj;

  int ix = (int)(floor(x));
  float xx = x - (float)(ix);
  int iy = (int)(floor(y));
  float yy = y - (float)(iy);
  int qx = (int)(xx * 5.0f);
  int qy = (int)(yy * 5.0f);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) {
      if (qx >= -2 + i * 2 + j && qx <= -2 + i * 2 + 1 + j &&
          qy >= 1 - i + 2 * j && qy <= 1 - i + 2 * j + 1) {
        for (ii = 0; ii <= 2; ii++)
          for (jj = 0; jj <= 2; jj++) {
            if (ii == 1 || jj == 1) {
              int rx = -2 + i * 2 + j - 3 + ii * 2 + jj;
              int ry = 1 - i + 2 * j - 1 + jj * 2 - ii;
              dx[nc] = 1.0f / 5.0f;
              dy[nc] = 1.0f / 5.0f;
              cx[nc] = (float)(ix) + (float)(rx) / 5.0f + 1.0f / 5.0f;
              cy[nc] = (float)(iy) + (float)(ry) / 5.0f + 1.0f / 5.0f;
              nc++;
            }
          }

        int rx = -2 + i * 2 + j;
        int ry = 1 - i + 2 * j;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(rx - 1) / 5.0f + 0.5f / 5.0f;
        cy[nc] = (float)(iy) + (float)(ry) / 5.0f + 0.5f / 5.0f;
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(rx) / 5.0f + 0.5f / 5.0f;
        cy[nc] = (float)(iy) + (float)(ry + 2) / 5.0f + 0.5f / 5.0f;
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(rx + 2) / 5.0f + 0.5f / 5.0f;
        cy[nc] = (float)(iy) + (float)(ry + 1) / 5.0f + 0.5f / 5.0f;
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(rx + 1) / 5.0f + 0.5f / 5.0f;
        cy[nc] = (float)(iy) + (float)(ry - 1) / 5.0f + 0.5f / 5.0f;
        nc++;

        return;
      }
    }

  for (i = 0; i < 3; i++)
    for (j = 0; j < 2; j++) {
      if (qx == i * 2 + j && qy == 2 + 2 * j - i) {
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx - 2) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy + 1) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx + 1) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy + 2) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx - 1) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy - 2) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 0.5f / 5.0f;
        dy[nc] = 0.5f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx + 2) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy - 1) / 5.0f + dy[nc];
        nc++;

        dx[nc] = 1.0f / 5.0f;
        dy[nc] = 1.0f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx - 2) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy - 1) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 1.0f / 5.0f;
        dy[nc] = 1.0f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx - 1) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy + 1) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 1.0f / 5.0f;
        dy[nc] = 1.0f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx + 1) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy) / 5.0f + dy[nc];
        nc++;
        dx[nc] = 1.0f / 5.0f;
        dy[nc] = 1.0f / 5.0f;
        cx[nc] = (float)(ix) + (float)(qx) / 5.0f + dx[nc];
        cy[nc] = (float)(iy) + (float)(qy - 2) / 5.0f + dy[nc];
        nc++;

        return;
      }
    }

  // if here, error in paveb

  return;
}

//------------------------------------------------------------------------------
// PP_pavement
//------------------------------------------------------------------------------
void PP_pavement(float x, float y, int tt, int decalx, int Nx, float correction,
                 float *ccx, float *ccy, float *cdx, float *cdy) {

  switch (tt) {
  case PP_tilingType_REGULAR:
    PP_pave(x, y, decalx, correction, 0.0, 0.0, ccx, ccy, cdx, cdy);
    break;

  case PP_tilingType_IRREGULAR:
    PP_pave(x, y, decalx, correction, 0.8, 0.8, ccx, ccy, cdx, cdy);
    break;

  case PP_tilingType_CROSS:
    PP_paved(x, y, Nx, ccx, ccy, cdx, cdy);
    break;

  case PP_tilingType_BISQUARE:
    PP_paveb(x, y, ccx, ccy, cdx, cdy);
    break;

  case PP_tilingType_IRREGULARX:
    PP_pave(x, y, decalx, correction, 0.8, 0.0, ccx, ccy, cdx, cdy);
    break;

  case PP_tilingType_IRREGULARY:
    PP_pave(x, y, decalx, correction, 0.0, 0.8, ccx, ccy, cdx, cdy);
    break;

  default:
    PP_pave(x, y, decalx, correction, 0.0, 0.0, ccx, ccy, cdx, cdy);
    break;
  }
}

//------------------------------------------------------------------------------
// PP_pointset
//------------------------------------------------------------------------------
int PP_pointset(
    // point set parameters
    float psubx, float psuby, float jitx, float jity, float *ccx, float *ccy,
    float *cdx, float *cdy, float *cx, float *cy, float *ncx, float *ncy,
    float *ndx, float *ndy) {

  int i, j, k;
  int nc = 0;

  for (k = 0; k < 9; k++) {
    int ix = (int)(floor(ccx[k]));
    float xx = ccx[k] - (float)(ix);
    int iy = (int)(floor(ccy[k]));
    float yy = ccy[k] - (float)(iy);
    seeding((uint)((int)(floor(ccx[k] * 15.0f)) + 10),
            (uint)((int)(floor(ccy[k] * 10.0f)) + 3), 0u);
    float subx = next() * 0.5f + 0.5f;
    // float suby = next() * 0.5f + 0.5f;
    float dif = cdx[k] - cdy[k];
    if (dif < 0.0f)
      dif = -dif;
    if (dif < 0.1 && (subx < psubx)) // || suby < psuby ) )
    {
      float cutx = 0.5f + 0.2 * next() * jitx;
      float cuty = 0.5f + 0.2 * next() * jity;
      float ncdx, ncdy, nccx, nccy, rx, ry;

      ncdx = (cutx * 2.0f * cdx[k]) * 0.5f;
      ncdy = (cuty * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + ncdx;
      nccy = ccy[k] - cdy[k] + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;

      ncdx = ((1.0f - cutx) * 2.0f * cdx[k]) * 0.5f;
      ncdy = (cuty * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + (cutx * 2.0f * cdx[k]) + ncdx;
      nccy = ccy[k] - cdy[k] + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;

      ncdx = (cutx * 2.0f * cdx[k]) * 0.5f;
      ncdy = ((1.0f - cuty) * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + ncdx;
      nccy = ccy[k] - cdy[k] + (cuty * 2.0f * cdy[k]) + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;

      ncdx = ((1.0f - cutx) * 2.0f * cdx[k]) * 0.5f;
      ncdy = ((1.0f - cuty) * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + (cutx * 2.0f * cdx[k]) + ncdx;
      nccy = ccy[k] - cdy[k] + (cuty * 2.0f * cdy[k]) + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;
    } else if (cdx[k] > cdy[k] + 0.1 && subx < psubx) {
      float cutx = 0.4 + 0.2 * (next() * 0.5f + 0.5f);
      float cuty = 1.0f;
      float ncdx, ncdy, nccx, nccy, rx, ry;

      ncdx = (cutx * 2.0f * cdx[k]) * 0.5f;
      ncdy = (cuty * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + ncdx;
      nccy = ccy[k] - cdy[k] + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;

      ncdx = ((1.0f - cutx) * 2.0f * cdx[k]) * 0.5f;
      ncdy = (cuty * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + (cutx * 2.0f * cdx[k]) + ncdx;
      nccy = ccy[k] - cdy[k] + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;
    } else if (cdy[k] > cdx[k] + 0.1 && subx < psuby) {
      float cutx = 1.0f;
      float cuty = 0.4 + 0.2 * (next() * 0.5f + 0.5f);
      float ncdx, ncdy, nccx, nccy, rx, ry;

      ncdx = (cutx * 2.0f * cdx[k]) * 0.5f;
      ncdy = (cuty * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + ncdx;
      nccy = ccy[k] - cdy[k] + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;

      ncdx = (cutx * 2.0f * cdx[k]) * 0.5f;
      ncdy = ((1.0f - cuty) * 2.0f * cdy[k]) * 0.5f;
      nccx = ccx[k] - cdx[k] + ncdx;
      nccy = ccy[k] - cdy[k] + (cuty * 2.0f * cdy[k]) + ncdy;
      rx = ncdx * next() * jitx;
      ry = ncdy * next() * jity;
      cx[nc] = nccx + rx;
      cy[nc] = nccy + ry;
      ncx[nc] = nccx;
      ncy[nc] = nccy;
      ndx[nc] = ncdx;
      ndy[nc] = ncdy;
      nc++;
    } else {
      float rx = cdx[k] * next() * jitx;
      float ry = cdy[k] * next() * jity;
      cx[nc] = ccx[k] + rx;
      cy[nc] = ccy[k] + ry;
      ncx[nc] = ccx[k];
      ncy[nc] = ccy[k];
      ndx[nc] = cdx[k];
      ndy[nc] = cdy[k];
      nc++;
    }
  }

  return nc;
}

//------------------------------------------------------------------------------
// PP_distribute
//------------------------------------------------------------------------------
int PP_distribute(
    // position
    float px, float py,
    // point set parameters
    int tt, float psubx, float psuby, int decalx, int Nx, float correction,
    float jitter, float *cx, float *cy, float *ncx, float *ncy, float *ndx,
    float *ndy) {

  float ccx[9];
  float ccy[9];
  float cdx[9];
  float cdy[9];

  PP_pavement(px, py, tt, decalx, Nx, correction, ccx, ccy, cdx, cdy);

  int np = PP_pointset(psubx, psuby, 0.9, 0.9, ccx, ccy, cdx, cdy, cx, cy, ncx,
                       ncy, ndx, ndy);

  for (int i = 0; i < np; i++) {
    cx[i] = cx[i] * jitter + ncx[i] * (1.0f - jitter);
    cy[i] = cy[i] * jitter + ncy[i] * (1.0f - jitter);
  }

  return np;
}

//------------------------------------------------------------------------------
// PP_genPointSet
//------------------------------------------------------------------------------
int PP_genPointSet(
    // position
    float x, float y,
    // point set parameters
    int pointsettype, float jitter, float *px, float *py, float *ncx,
    float *ncy, float *ndx, float *ndy) {

  int tt;
  float ppointsub;
  int decalx;
  int Nx;

  float correction = 0.0f;

  switch (pointsettype) {
  case 0:
    tt = PP_tilingType_REGULAR;
    ppointsub = 0.0f;
    decalx = 1;
    Nx = 0;
    break;

  case 1:
    tt = PP_tilingType_REGULAR;
    ppointsub = 0.5f;
    decalx = 1;
    Nx = 0;
    break;

  case 2:
    tt = PP_tilingType_REGULAR;
    ppointsub = 0.0f;
    decalx = 2;
    Nx = 0;
    correction = 0.25f;
    break;

  case 3:
    tt = PP_tilingType_REGULAR;
    ppointsub = 0.0f;
    decalx = 3;
    Nx = 0;
    correction = 0.25f;
    break;

  case 4:
    tt = PP_tilingType_IRREGULAR;
    ppointsub = 0.0f;
    decalx = 1;
    Nx = 0;
    break;

  case 5:
    tt = PP_tilingType_IRREGULAR;
    ppointsub = 0.5f;
    decalx = 1;
    Nx = 0;
    break;

  case 6:
    tt = PP_tilingType_IRREGULARX;
    ppointsub = 0.0f;
    decalx = 1;
    Nx = 0;
    break;

  case 7:
    tt = PP_tilingType_IRREGULARX;
    ppointsub = 0.5f;
    decalx = 1;
    Nx = 0;
    break;

  case 8:
    tt = PP_tilingType_CROSS;
    ppointsub = 0.0f;
    decalx = 0;
    Nx = 2;
    break;

  case 9:
    tt = PP_tilingType_CROSS;
    ppointsub = 0.5f;
    decalx = 0;
    Nx = 2;
    break;

  case 10:
    tt = PP_tilingType_CROSS;
    ppointsub = 0.0f;
    decalx = 0;
    Nx = 3;
    break;

  case 11:
    tt = PP_tilingType_CROSS;
    ppointsub = 0.5f;
    decalx = 0;
    Nx = 3;
    break;

  case 12:
    tt = PP_tilingType_BISQUARE;
    ppointsub = 0.0f;
    decalx = 0;
    Nx = 1;
    break;

  case 13:
    tt = PP_tilingType_BISQUARE;
    ppointsub = 0.5f;
    decalx = 0;
    Nx = 1;
    break;

  default:
    tt = PP_tilingType_REGULAR;
    ppointsub = 0.0f;
    decalx = 1;
    Nx = 0;
    break;
  }

  // Compute points
  return PP_distribute(x, y, tt, ppointsub, ppointsub, decalx, Nx, correction,
                       jitter, px, py, ncx, ncy, ndx, ndy);
}

//------------------------------------------------------------------------------
// PP_cdistance
//------------------------------------------------------------------------------

float PP_cdistance( float x1, float y1, float x2, float y2, float norm, 
float cx, float cy, float dx, float dy, float larp )
{
	float ddx = ( x1 - x2 );
	float ddy = ( y1 - y2 );
	float ex = ddx < 0.0f ? (-ddx) / ( x2 - cx + dx ): ddx / ( cx + dx - x2 );
	float ey = ddy < 0.0f ? (-ddy) / ( y2 - cy + dy ) : ddy / ( cy + dy - y2 );
	return ((1.0f-larp)*pow(pow(fabs(ddx), norm) + pow(fabs(ddy), norm), 1.0f / norm)+larp*(ex > ey ? ex : ey));
}

//------------------------------------------------------------------------------
// PP_nthclosest
//------------------------------------------------------------------------------

void PP_nthclosest(int *mink, int nn, float xx, float yy, float *cx, float *cy,
                   int nc, float *ncx, float *ncy, float *dx, float *dy, float larp) {
  int i, k;

  float dist[36];

  for (k = 0; k < nc; k++) {
    float dd = PP_cdistance(xx, yy, cx[k], cy[k], 2.0, ncx[k], ncy[k], dx[k], dy[k], larp);
    dist[k] = dd;
  }

  for (i = 0; i < nn; i++) {
    int mk = 0;
    for (k = 1; k < nc; k++) {
      if (dist[mk] > dist[k])
        mk = k;
    }
    mink[i] = mk;
    dist[mk] = 100000.0f;
  }

  // Pad the remaining of the mink array with -1
  // (required for jax procedural_pptbf_2 function)
  for (i = nn; i < 36; i++) {
    mink[i] = -1;
  }
}

//------------------------------------------------------------------------------
// interTriangle
//------------------------------------------------------------------------------
float interTriangle(float origx, float origy, float ddx, float ddy,
                    float startx, float starty, float endx, float endy) {

  float dirx = (endx - startx);
  float diry = (endy - starty);
  float dirno = sqrt(dirx * dirx + diry * diry);
  dirx /= dirno;
  diry /= dirno;
  float val = ddx * diry - ddy * dirx;
  float segx = -(startx - origx);
  float segy = -(starty - origy);
  float lambda = (dirx * segy - diry * segx) / val;

  return lambda;
}

//------------------------------------------------------------------------------
// bezier2
//------------------------------------------------------------------------------
float2 bezier2(float ts, float p0x, float p0y, float p1x, float p1y, float p2x,
               float p2y) {

  float p01x = ts * p1x + (1.0f - ts) * p0x;
  float p01y = ts * p1y + (1.0f - ts) * p0y;
  float p11x = ts * p2x + (1.0f - ts) * p1x;
  float p11y = ts * p2y + (1.0f - ts) * p1y;

  float2 spline =
      (float2)(ts * p11x + (1.0f - ts) * p01x, ts * p11y + (1.0f - ts) * p01y);

  return spline;
}

float intersectRays(float2 P, float2 r, float2 Q, float2 s) {
  //# Ray1 = P + lambda1 * r
  //# Ray2 = Q + lambda2 * s
  //# r and s must be normalized (length = 1)
  //# Returns intersection point along Ray1
  //# Returns null if lines do not intersect or are identical

  float PQx = Q.x - P.x;
  float PQy = Q.y - P.y;
  float rxt = -r.y;
  float ryt = r.x;
  float qx = PQx * r.x + PQy * r.y;
  float qy = PQx * rxt + PQy * ryt;
  float ssx = s.x * r.x + s.y * r.y;
  float ssy = s.x * rxt + s.y * ryt;
  // If rays are identical or do not cross
  if (ssy == 0.0f)
    return NAN;

  float a = qx - qy * ssx / ssy;
  return a;
}

/*
//------------------------------------------------------------------------------
// celldist (NO LARP)
//------------------------------------------------------------------------------

float2 circumcenter(float ax, float ay, float bx, float by, float cx,
                    float cy) {

  float d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  float ux =
      ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) +
       (cx * cx + cy * cy) * (ay - by)) /
      d;
  float uy =
      ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) +
       (cx * cx + cy * cy) * (bx - ax)) /
      d;
  return (float2)(ux, uy);
}

bool isDelaunay(int i, int *mink, float *cx, float *cy, int nc) {
  bool delaunay = true;
  // Check if Gabriel
  // Edge middle and length
  float midX = 0.5f * (cx[mink[0]] + cx[mink[i]]);
  float midY = 0.5f * (cy[mink[0]] + cy[mink[i]]);
  float dx = cx[mink[i]] - cx[mink[0]];
  float dy = cy[mink[i]] - cy[mink[0]];
  float radg = 0.25 * (dx * dx + dy * dy);
  // Only the feature points at a distance that is less than
  // the distance to the i - th feature point have to be checked
  int k = 1;
  while (k < i && delaunay) {
    dx = cx[mink[k]] - midX;
    dy = cy[mink[k]] - midY;
    float dist = dx * dx + dy * dy;
    if (dist < radg) {
      // Not Gabriel : check if Delaunay
      // Triangle circumcenter and radius
      float2 u = circumcenter(cx[mink[0]], cy[mink[0]], cx[mink[i]],
                              cy[mink[i]], cx[mink[k]], cy[mink[k]]);
      dx = cx[mink[k]] - u.x;
      dy = cy[mink[k]] - u.y;
      float radd = dx * dx + dy * dy;
      // Delaunay test : all the feature points have to
      // be tested for empty disk
      for (int l = 1; l < nc; l++) {
        if (l != i && l != k) {
          dx = cx[mink[l]] - u.x;
          dy = cy[mink[l]] - u.y;
          dist = dx * dx + dy * dy;
          if (dist < radd) {
            delaunay = false;
            break;
          }
        }
      }
    }
    k++;
  }
  return delaunay;
}

void PP_delaunay(int nc, float x, float y, int *mink, float *cx, float *cy,
                 bool *delaunay) {
  for (int i = 1; i < nc; i++) {
    delaunay[i] = isDelaunay(i, mink, cx, cy, nc);
  }
}

float celldist(float ixx, float iyy, int *mink, bool *delaunay, float *cx,
               float *cy, int nc) {
  // Compute closest feature point for (ixx, iyy)
  float2 closest1 = (float2)(cx[mink[0]], cy[mink[0]]);

  // Ray from closest1 to(ixx, iyy)
  float2 dir = (float2)(ixx - closest1.x, iyy - closest1.y);
  float2 dirNorm = normalize(dir);

  float minDist = 10000.0f;
  float2 closest2;
  float2 dirBisec, dirBisecNorm;
  float2 mid;
  float2 dirEdgeNorm;
  float a;

  for (int i = 1; i < nc; i++) {
    closest2 = (float2)(cx[mink[i]], cy[mink[i]]);

    // Edge betwen closest1 and closest2
    dirBisec = closest2 - closest1;

    // If closest 2 is in the good half plane
    if (dot(dirNorm, dirBisec) >= 0.0f) {
      if (delaunay[i]) {

        mid = 0.5f * (closest1 + closest2);
        dirBisecNorm = normalize(dirBisec);
        dirEdgeNorm = (float2)(-dirBisecNorm.y, dirBisecNorm.x);

        a = intersectRays(closest1, dirNorm, mid, dirEdgeNorm);

        if (a < minDist) {
          minDist = a;
        }
      }
    }
  }

  return minDist;
}
*/

//------------------------------------------------------------------------------
// celldist
//------------------------------------------------------------------------------

int cclosest( float xx, float yy, float cx[9 * 4], float cy[9 * 4], int nc, 
float cnx[9 * 4], float cny[9 * 4], float dx[9 * 4], float dy[9 * 4], float larp )
{
	int mink = 0;
	float mind = 0.0f;
	int k;
	for ( k = 0; k < nc; k++ )
	{
		float dd = PP_cdistance( xx, yy, cx[ k ], cy[ k ], 2.0, cnx[ k ], cny[ k ],
    dx[ k ], dy[ k ], larp );
		if ( k == 0 ) { mind = dd; }
		else if ( mind > dd ) { mink = k; mind = dd; }
	}
	
	return mink;
}

float celldist( float ixx, float iyy, int k, int mink, float cx[9 * 4], float cy[9 * 4], int nc,
float cnx[9 * 4], float cny[9 * 4], float dx[9 * 4], float dy[9 * 4], float larp )
{
	float delta = 0.2f;
	int count, nk;
	float xx, yy, ddx, ddy, dd;
	
	do
	{
		xx = ixx; yy = iyy;
		ddx = cx[ k ] - xx; ddy = cy[ k ] - yy;
		dd = sqrt( ddx * ddx + ddy * ddy );
		if ( dd < 0.001f ) return 0.0f;
		ddx *= delta / dd; ddy *= delta / dd;
		if ( k == mink ) { ddx = -ddx; ddy = -ddy; }
		count = 0;
		do
		{
			xx += ddx; yy += ddy;
			nk = cclosest( xx, yy, cx, cy, nc, cnx, cny, dx, dy, larp );
			count++;
		} while ( ( ( k == mink && nk == k ) || ( k != mink && nk != k ) ) && count < 100 );
		delta /= 2.0f;
	} while ( count == 100 && delta >= 0.01f );
	float xa = xx - ddx, ya = yy - ddy;
	float midx = ( xa + xx ) / 2.0f, midy = ( ya + yy ) / 2.0f;
	for ( int i = 0; i < 5; i++ )
	{
		nk = cclosest( midx, midy, cx, cy, nc, cnx, cny,dx, dy, larp );
		if ( ( ( k == mink && nk == k ) || ( k != mink && nk != k ) ) ) { xa = midx; ya = midy; }
		else { xx = midx; yy = midy; }
		midx = ( xa + xx ) / 2.0f; midy = ( ya + yy ) / 2.0f;
	}
	float vdx = cx[k] - midx, vdy = cy[k] - midy;
	return sqrt(vdx*vdx + vdy*vdy);
}

//------------------------------------------------------------------------------
// PPTBF
//------------------------------------------------------------------------------

float procedural_pptbf(float xx, float yy, float tx, float ty, float zoom,
                       float alpha, int tt, float jitter, float arity,
                       int ismooth, float wsmooth, float normblend,
                       float normsig, float larp, float normfeat, float winfeatcorrel,
                       float feataniso, float sigcos, float deltaorient,
                       float amp, float rx, float ry) {

  float pptbf = 0.0f;

  // Translation

  float ppx = xx + tx;
  float ppy = yy + ty;

  // Deformation

  ppx = ppx + amp * cnoise2DG(ppx * zoom * 0.5f + rx, ppy * zoom * 0.5f) +
        amp * 0.5f * cnoise2DG(ppx * zoom + rx, ppy * zoom) +
        amp * 0.25f * cnoise2DG(ppx * zoom * 2.0f + rx, ppy * zoom * 2.0f) +
        amp * 0.125f * cnoise2DG(ppx * zoom * 4.0f + rx, ppy * zoom * 4.0f);

  ppy = ppy + amp * cnoise2DG(ppx * zoom * 0.5f, ppy * zoom * 0.5f + ry) +
        amp * 0.5f * cnoise2DG(ppx * zoom, ppy * zoom + ry) +
        amp * 0.25f * cnoise2DG(ppx * zoom * 2.0f, ppy * zoom * 2.0f + ry) +
        amp * 0.125f * cnoise2DG(ppx * zoom * 4.0f, ppy * zoom * 4.0f + ry);

  /*float freq = 3.0f;

  ppx = ppx +
        amp * 0.25f *
            cnoise2DG(ppx * zoom * freq * 4.0f + rx, ppy * zoom * freq * 4.0f) +
        amp * 0.125f *
            cnoise2DG(ppx * zoom * freq * 8.0f + rx, ppy * zoom * freq * 8.0f);

  ppy = ppy +
        amp * 0.25f *
            cnoise2DG(ppx * zoom * freq * 2.0f, ppy * zoom * freq * 4.0f + ry) +
        amp * 0.125f *
            cnoise2DG(ppx * zoom * freq * 8.0f, ppy * zoom * freq * 8.0f +
  ry);*/

  // Model Transform

  // Note: +100.0 is required to avoir negative coordinates for the PRNG!
  float x = 100.0 + (ppx * cos(-alpha) + ppy * sin(-alpha)) * zoom;
  float y = 100.0 + (-ppx * sin(-alpha) + ppy * cos(-alpha)) * zoom;

  //  Point Process

  // Feature points locations with jittering
  float px[36];
  float py[36];

  // Feature points locations without jittering (i.e.tile centers)
  float ncx[36];
  float ncy[36];

  // Distance to cell borders
  float ndx[36];
  float ndy[36];

  // Closest neighbors indices
  int mink[36];

  // Delaunay edges
  //bool delaunay[36];

  int nc = PP_genPointSet(x, y, tt, jitter, px, py, ncx, ncy, ndx, ndy);

  // Maximum number of closest neighbors
  int npp = (nc < PP_nbMaxNeighbors ? nc : PP_nbMaxNeighbors);

  // Code adapted for norm = 2
  // For other norms, see:
  // Two - Dimensional Voronoi Diagrams in the Lp - Metric
  // https: //
  // www.semanticscholar.org/paper/Two-Dimensional-Voronoi-Diagrams-in-the-Lp-Metric-Lee/ff282c65cf3c5cd0f0d87523d9f83ecae6510fcc

  // Compute the table of npp closest feature points from(x, y)
  // Note : the max value of npp is typically set to 18 which may be less than
  // nc (nc may reach 30 e.g.with tiling type 1)

  PP_nthclosest(mink, npp, x, y, px, py, nc, ncx, ncy, ndx, ndy, larp);
  
  //PP_delaunay(npp, x, y, mink, px, py, delaunay);

  // PPTBF = PP x ( W F )

  float vv = 0.0f;

  for (int k = 0; k < npp; k++) {

    seeding((uint)(px[mink[k]] * 12.0f + 7.0f),
            (uint)(py[mink[k]] * 12.0f + 1.0f), 0u);

    //float dalpha = 2.0f * M_PI_F / pow(2.0f, (float)((uint)(arity + 0.5f)));
    float dalpha = 2.0 * M_PI_F / 8.0;
    //float rotalpha = dalpha * (next() * 0.5f + 0.5f);
    float rotalpha = dalpha * 0.5f;

    // Window Function: W

    float ddx = (x - px[mink[k]]);
    float ddy = (y - py[mink[k]]);

    // Distance to current feature point
    float sdd = sqrt(ddx * ddx + ddy * ddy);

    /*
    // Uncomment to display the point set
    if (sdd < 0.01)
      return 1.0;
    else
      return 0.0;
    */

    float gauss = 1.0f;
    float footprint = 1.0f;

    // Reduced footprint for some tiling types
    if (tt >= 8) {
      footprint *= 0.4f;
    }

    gauss = (exp(-2.0f * sdd) - exp(-2.0f * footprint)) /
            (1.0f - exp(-2.0f * footprint));

    if (gauss < 0.0f) {
      gauss = 0.0f;
    } else if (gauss > 1.0f) {
      gauss = 1.0f;
    }

    // Cellular Window

    float cv = 0.0f;

    if (k == 0 && sdd < 0.0001f) {
      cv = 1.0f;
    } else if (k == 0) {
      // k == 0 : closest feature point
      // sdd : distance to this feature point

      // ---- Begin smoothing ----

      float celldd0 = celldist( x, y, mink[k], mink[0], px, py, nc, ncx, ncy, ndx, ndy, larp );

      ddx /= sdd;
      ddy /= sdd;

      // atan2 returns an angle in the range [-pi, pi]
      float alpha = atan2(-ddy, -ddx) + M_PI_F;
      float palpha = alpha - rotalpha + 2.0f * M_PI_F;

      int ka = (int)(palpha / dalpha);
      
      float ptx = px[mink[0]] + 0.1f * cos(dalpha * (float)(ka) + rotalpha);
      float pty = py[mink[0]] + 0.1f * sin(dalpha * (float)(ka) + rotalpha);
      //float celldd1 = celldist(ptx, pty, mink, delaunay, px, py, npp);
      float celldd1 = celldist(ptx, pty, mink[k], mink[0], px, py, nc, ncx, ncy, ndx, ndy, larp);
      float startx =
          px[mink[0]] + celldd1 * cos(dalpha * (float)(ka) + rotalpha);
      float starty =
          py[mink[0]] + celldd1 * sin(dalpha * (float)(ka) + rotalpha);
      
      ptx = px[mink[0]] + 0.1f * cos(dalpha * (float)(ka) + dalpha + rotalpha);
      pty = py[mink[0]] + 0.1f * sin(dalpha * (float)(ka) + dalpha + rotalpha);
      //float celldd2 = celldist(ptx, pty, mink, delaunay, px, py, npp);
      float celldd2 = celldist(ptx, pty, mink[k], mink[0], px, py, nc, ncx, ncy, ndx, ndy, larp);
      float endx =
          px[mink[0]] + celldd2 * cos(dalpha * (float)(ka) + dalpha + rotalpha);
      float endy =
          py[mink[0]] + celldd2 * sin(dalpha * (float)(ka) + dalpha + rotalpha);

      float midx = (startx + endx) / 2.0f;
      float midy = (starty + endy) / 2.0f;

      float sa = (px[mink[0]] - x) * (midy - py[mink[0]]) - (midx - px[mink[0]] ) * (py[mink[0]] - y);

      float2 spline;
      float2 smooth;

      if (sa > 0.0) {
        ptx = px[mink[0]] +
              0.1f * cos(dalpha * (float)(ka) + 2.0f * dalpha + rotalpha);
        pty = py[mink[0]] +
              0.1f * sin(dalpha * (float)(ka) + 2.0f * dalpha + rotalpha);
        //float celldd = celldist(ptx, pty, mink, delaunay, px, py, npp);
        float celldd = celldist(ptx, pty, mink[k], mink[0], px, py, nc, ncx, ncy, ndx, ndy, larp);

        float nendx = px[mink[0]] + celldd * cos(dalpha * (float)(ka) +
                                                 2.0f * dalpha + rotalpha);
        float nendy = py[mink[0]] + celldd * sin(dalpha * (float)(ka) +
                                                 2.0f * dalpha + rotalpha);
        nendx = (nendx + endx) / 2.0f;
        nendy = (nendy + endy) / 2.0f;

        float lambda = interTriangle(px[mink[0]], py[mink[0]], ddx, ddy, midx,
                                     midy, nendx, nendy);
        
        float bordx = ddx * lambda + px[mink[0]];
        float bordy = ddy * lambda + py[mink[0]];
        float dirno = sqrt((nendx - midx) * (nendx - midx) +
                           (nendy - midy) * (nendy - midy));
        float ts = sqrt((bordx - midx) * (bordx - midx) +
                        (bordy - midy) * (bordy - midy));
        ts /= dirno;
        spline = bezier2(ts, midx, midy, endx, endy, nendx, nendy);
        smooth.x = bordx;
        smooth.y = bordy;
      } else {
        ptx = px[mink[0]] + 0.1f * cos(dalpha * (float)(ka)-dalpha + rotalpha);
        pty = py[mink[0]] + 0.1f * sin(dalpha * (float)(ka)-dalpha + rotalpha);
        //float celldd = celldist(ptx, pty, mink, delaunay, px, py, npp);
        float celldd = celldist(ptx, pty, mink[k], mink[0], px, py, nc, ncx, ncy, ndx, ndy, larp);

        float nstartx =
            px[mink[0]] + celldd * cos(dalpha * (float)(ka)-dalpha + rotalpha);
        float nstarty =
            py[mink[0]] + celldd * sin(dalpha * (float)(ka)-dalpha + rotalpha);
        nstartx = (nstartx + startx) / 2.0f;
        nstarty = (nstarty + starty) / 2.0f;

        float lambda = interTriangle(px[mink[0]], py[mink[0]], ddx, ddy,
                                     nstartx, nstarty, midx, midy);
        float bordx = ddx * lambda + px[mink[0]];
        float bordy = ddy * lambda + py[mink[0]];
        float dirno = sqrt((midx - nstartx) * (midx - nstartx) +
                           (midy - nstarty) * (midy - nstarty));
        float ts = sqrt((bordx - nstartx) * (bordx - nstartx) +
                        (bordy - nstarty) * (bordy - nstarty));
        ts /= dirno;
        spline = bezier2(ts, nstartx, nstarty, startx, starty, midx, midy);
        smooth.x = bordx;
        smooth.y = bordy;
      }

      /*float lambda = interTriangle(px[mink[0]], py[mink[0]], ddx, ddy, startx,
                                   starty, endx, endy);

      float smoothdist =
          sqrt((smooth.x - px[mink[0]]) * (smooth.x - px[mink[0]]) +
               (smooth.y - py[mink[0]]) * (smooth.y - py[mink[0]]));
      */

      float splinedist =
          sqrt((spline.x - px[mink[0]]) * (spline.x - px[mink[0]]) +
               (spline.y - py[mink[0]]) * (spline.y - py[mink[0]]));

      /*if (ismooth == 0) {
        cv = (1.0f - wsmooth) * (1.0f - sdd / lambda) +
             wsmooth * (1.0f - sdd / smoothdist);
      } else {
        cv = (1.0f - wsmooth) * (1.0f - sdd / smoothdist) +
             wsmooth * (1.0f - sdd / splinedist);
      }*/

      cv = (1.0f - wsmooth) * (1.0f - sdd / celldd0) +
           wsmooth * (1.0f - sdd / splinedist);

      // ---- End Smoothing ----
      

      if (cv < 0.0f) {
        cv = 0.0f;
      } else if (cv > 1.0f) {
        cv = 1.0f;
      }
    }

    // For smoothing debugging
    //return cv;

    float coeff1 = normblend *
                   (exp((cv - 1.0f) * normsig) - exp(-1.0f * normsig)) /
                   (1.0f - exp(-1.0f * normsig));
    float coeff2 = (1.0f - normblend) * gauss;

    float winsum = coeff1 + coeff2;

    // Feature function

    float feat = 0.0f;

    {
      // Gabor
      seeding((uint)(px[mink[k]] * 15.0f + 2.0f),
              (uint)(py[mink[k]] * 15.0f + 5.0f), 0u);

      float lx = ncx[mink[k]] + next() * 0.99f * ndx[mink[k]];
      float ly = ncy[mink[k]] + next() * 0.99f * ndy[mink[k]];
      lx = winfeatcorrel * px[mink[k]] + (1.0f - winfeatcorrel) * lx;
      ly = winfeatcorrel * py[mink[k]] + (1.0f - winfeatcorrel) * ly;
      float deltalx = (x - lx) / ndx[mink[k]];
      float deltaly = (y - ly) / ndy[mink[k]];
      float angle = deltaorient * next();
      float ddx = (deltalx * cos(-angle) + deltaly * sin(-angle));
      float iddy = (-deltalx * sin(-angle) + deltaly * cos(-angle));
      float ddy = iddy / pow(2.0f, feataniso);
      //float ddy = iddy * (feataniso * 0.05 + (1.0 - feataniso));
      float dd2 = pow(pow(fabs(ddx), normfeat) + pow(fabs(ddy), normfeat),
                      1.0f / normfeat);
      if (normfeat > 2.0f) {
        dd2 = (normfeat - 2.0f) *
                  (fabs(ddx) > fabs(ddy) ? fabs(ddx) : fabs(ddy)) +
              (1.0f - (normfeat - 2.0f)) * dd2;
      }
      // float ddist = dd2 / (footprint / sigcos);
      float ddist = (sigcos * dd2) / footprint;

      feat = 0.5f * exp(-ddist);
    }

    vv += winsum * feat;
    //vv += winsum;
  }

  if (vv < 0.0f) {
    vv = 0.0f;
  }

  pptbf = vv;

  return pptbf;
}

__kernel void pptbf(const uint size, float tx, float ty, float zoom,
                    float alpha, int tt, float jitter, float arity, int ismooth,
                    float wsmooth, float normblend, float normsig, float larp,
                    float normfeat, float winfeatcorrel, float feataniso,
                    float sigcos, float deltaorient, float amp, float rx,
                    float ry, __global __write_only float *image_g) {

  int i = get_global_id(0);
  int j = get_global_id(1);

  float x = (float)j / (float)size;
  float y = (float)i / (float)size;

  // printf("%d %d\n", i, j);

  /*
  image_g[i * size + j] = procedural_pptbf(x, y,
                                           0.0,   // tx
                                           0.0,   // ty
                                           5,     // zoom
                                           0.0,   // alpha
                                           0,     // tiling type
                                           0.01,  // jitter
                                           2,     // arity
                                           0,     // ismooth
                                           0.5,   // wsmooth
                                           0.8,   // normblend
                                           0.01,  // normsig
                                           0.5,   // larp
                                           2,     // normfeat
                                           0,     // winfeatcorrel
                                           5,     // feataniso
                                           0.855, // sigcos
                                           1.0);  // deltaorient
  */

  image_g[i * size + j] =
      procedural_pptbf(x, y, tx, ty, zoom, alpha, tt, jitter, arity, ismooth,
                       wsmooth, normblend, normsig, larp, normfeat, winfeatcorrel,
                       feataniso, sigcos, deltaorient, amp, rx, ry);
}

/*
float sigmoid(float x, float t) {
  return (1.0f/(1.0f + exp(-60.0f*(x - t))));
}

__kernel void pptbf_smooth_thresh(const uint size, float tx, float ty, float zoom,
                                  float alpha, int tt, float jitter, float arity, int ismooth,
                                  float wsmooth, float normblend, float normsig,
                                  float normfeat, float winfeatcorrel, float feataniso,
                                  float sigcos, float deltaorient, float amp,
                                  float rx, float ry, float t,
                                  __global __write_only float *image_g) {

  int i = get_global_id(0);
  int j = get_global_id(1);

  float x = (float)j / (float)size;
  float y = (float)i / (float)size;

  image_g[i * size + j] =
      procedural_pptbf(x, y, tx, ty, zoom, alpha, tt, jitter, arity, ismooth,
                       wsmooth, normblend, normsig, normfeat, winfeatcorrel,
                       feataniso, sigcos, deltaorient, amp, rx, ry);

  // Note: to apply the sigmoid function, the pptbf values should range
  // between 0 and 1.
}
*/

__kernel void equalize(__global const float *uniform,
                       __global const int *uniform_argsorted,
                       __global const float *image_flat,
                       __global const int *image_flat_argsorted,
                       __global float *image_flat_matched) {

  int i = get_global_id(0);
  image_flat_matched[i] = uniform[uniform_argsorted[image_flat_argsorted[i]]];
}

__kernel void thresh(const uint size, __global const float *image_g,
                     const float t, __global int *image_t_g) {

  int i = get_global_id(0);
  int j = get_global_id(1);

  if (image_g[i * size + j] > t) {
    image_t_g[i * size + j] = 255;
  } else {
    image_t_g[i * size + j] = 0;
  }
}

__kernel void mean_curvature_flow(const uint size,
                                  __global const float *image_g,
                                  __global float *image_s_g) {

  // After:
  // https://gitlab.gnome.org/GNOME/gegl/-/blob/master/operations/common/mean-curvature-blur.c
  // https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization

  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
    image_s_g[i * size + j] = image_g[i * size + j];
  } else {
    float center = image_g[i * size + j];
    float left = image_g[i * size + (j - 1)];
    float right = image_g[i * size + (j + 1)];
    float top = image_g[(i - 1) * size + j];
    float bottom = image_g[(i + 1) * size + j];
    float topleft = image_g[(i - 1) * size + (j - 1)];
    float topright = image_g[(i - 1) * size + (j + 1)];
    float bottomleft = image_g[(i + 1) * size + (j - 1)];
    float bottomright = image_g[(i + 1) * size + (j + 1)];

    float dx = right - left;
    float dy = bottom - top;
    float magnitude = sqrt(dx * dx + dy * dy);

    image_s_g[i * size + j] = center;

    if (magnitude > 1e-6) {
      float dx2 = dx * dx;
      float dy2 = dy * dy;
      float dxx = right + left - 2.0f * center;
      float dyy = bottom + top - 2.0f * center;
      float dxy = 0.25f * (bottomright - topright - bottomleft + topleft);
      float n = dx2 * dyy + dy2 * dxx - 2.0f * dx * dy * dxy;
      float d = sqrt(pow(dx2 + dy2, 3.0f));
      float mean_curvature = n / d;

      image_s_g[i * size + j] += (0.25f * magnitude * mean_curvature);
      
      if (image_s_g[i * size + j] < 0.0f)
        image_s_g[i * size + j] = 0.0f;
      if (image_s_g[i * size + j] > 1.0f)
        image_s_g[i * size + j] = 1.0f;
    }
  }
}

__kernel void mean_curvature(const uint size,
                             __global const float *image_g,
                             __global float *image_s_g) {

  // After:
  // https://gitlab.gnome.org/GNOME/gegl/-/blob/master/operations/common/mean-curvature-blur.c
  // https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization

  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
    //image_s_g[i * size + j] = image_g[i * size + j];
    image_s_g[i * size + j] = 0.0;
  } else {
    float center = image_g[i * size + j];
    float left = image_g[i * size + (j - 1)];
    float right = image_g[i * size + (j + 1)];
    float top = image_g[(i - 1) * size + j];
    float bottom = image_g[(i + 1) * size + j];
    float topleft = image_g[(i - 1) * size + (j - 1)];
    float topright = image_g[(i - 1) * size + (j + 1)];
    float bottomleft = image_g[(i + 1) * size + (j - 1)];
    float bottomright = image_g[(i + 1) * size + (j + 1)];

    float dx = right - left;
    float dy = bottom - top;
    float magnitude = sqrt(dx * dx + dy * dy);

    //image_s_g[i * size + j] = magnitude;

    if (magnitude > 0.001) {
      float dx2 = dx * dx;
      float dy2 = dy * dy;
      float dxx = right + left - 2.0f * center;
      float dyy = bottom + top - 2.0f * center;
      float dxy = 0.25f * (bottomright - topright - bottomleft + topleft);
      float n = dx2 * dyy + dy2 * dxx - 2.0f * dx * dy * dxy;
      float d = sqrt(pow(dx2 + dy2, 3.0f));
      float mean_curvature = n / d;
      image_s_g[i * size + j] = 0.25f * magnitude * mean_curvature;
    } else {
      image_s_g[i * size + j] = 0.0f;
    }
  }
}