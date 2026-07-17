#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_N 100
#define NFIELDS 4
#define NSIZES 6

/* Stockmayer parameters (SPC/E mapping) */
static const double SIGMA = 1.0;
static const double EPSILON = 1.0;
static const double MU_STAR = 3.16;
static const double KB_RED = 1.0;

/* Simulation grid */
static const int sizes[NSIZES] = {10, 20, 30, 50, 75, 100};
static const double fields_SI[NFIELDS] = {0.0, 1e8, 5e8, 1e9};

typedef struct { double x, y, z; } Vec3;
typedef struct { double ux, uy, uz; } Dip;

static Vec3 pos[MAX_N];
static Dip  dip[MAX_N];
static double T_red, E_red;
static int N;

double randf(void) { return (double)rand() / RAND_MAX; }
double randf2(void) { return 2.0*randf() - 1.0; }

double pair_energy(int i, int j) {
    double dx = pos[i].x - pos[j].x;
    double dy = pos[i].y - pos[j].y;
    double dz = pos[i].z - pos[j].z;
    double r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < 0.5) return 1e10;
    double r = sqrt(r2);
    double r6 = r2*r2*r2;
    double lj = 4.0*(1.0/(r6*r6) - 1.0/r6);
    double r3 = r*r2;
    double udotu = dip[i].ux*dip[j].ux + dip[i].uy*dip[j].uy + dip[i].uz*dip[j].uz;
    double uidotr = (dip[i].ux*dx + dip[i].uy*dy + dip[i].uz*dz)/r;
    double ujdotr = (dip[j].ux*dx + dip[j].uy*dy + dip[j].uz*dz)/r;
    double dd = MU_STAR*MU_STAR*(udotu/r3 - 3.0*uidotr*ujdotr/r3);
    return lj + dd;
}

double total_energy(void) {
    double E = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++)
            E += pair_energy(i, j);
        E -= E_red * dip[i].uz * MU_STAR;
    }
    return E;
}

double single_energy(int k) {
    double E = 0;
    for (int j = 0; j < N; j++)
        if (j != k) E += pair_energy(k, j);
    E -= E_red * dip[k].uz * MU_STAR;
    return E;
}

void init_config(void) {
    double box = pow(N, 1.0/3.0) * 1.2;
    for (int i = 0; i < N; i++) {
        pos[i].x = randf2() * box;
        pos[i].y = randf2() * box;
        pos[i].z = randf2() * box;
        double theta = acos(randf2());
        double phi = 2*M_PI*randf();
        dip[i].ux = sin(theta)*cos(phi);
        dip[i].uy = sin(theta)*sin(phi);
        dip[i].uz = cos(theta);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: mc_stockmayer T_K\n"); return 1; }
    double T_K = atof(argv[1]);
    srand(time(NULL) ^ (int)(T_K*1000));
    
    int neq_base = 3000, nprod_base = 5000;
    double dr = 0.08, dtheta = 0.20;
    
    for (int si = 0; si < NSIZES; si++) {
        N = sizes[si];
        for (int fi = 0; fi < NFIELDS; fi++) {
            T_red = T_K / 78.2;
            E_red = fields_SI[fi] * 3.166e-10 * 1.602e-19 / (78.2 * 1.381e-23);
            
            init_config();
            int neq = neq_base + 20*N;
            int nprod = nprod_base + 30*N;
            int accept = 0, total = 0;
            
            /* Equilibration */
            for (int step = 0; step < neq; step++) {
                for (int m = 0; m < N; m++) {
                    int k = rand() % N;
                    double e_old = single_energy(k);
                    Vec3 oldp = pos[k]; Dip oldd = dip[k];
                    if (randf() < 0.5) {
                        pos[k].x += randf2()*dr;
                        pos[k].y += randf2()*dr;
                        pos[k].z += randf2()*dr;
                    } else {
                        double th = dtheta*randf2(), ph = 2*M_PI*randf();
                        double ct=cos(th), st=sin(th), cp=cos(ph), sp=sin(ph);
                        double ux=dip[k].ux, uy=dip[k].uy, uz=dip[k].uz;
                        dip[k].ux = ux*ct + st*cp;
                        dip[k].uy = uy*ct + st*sp;
                        dip[k].uz = uz*ct;
                        double norm = sqrt(dip[k].ux*dip[k].ux+dip[k].uy*dip[k].uy+dip[k].uz*dip[k].uz);
                        if (norm > 0) { dip[k].ux/=norm; dip[k].uy/=norm; dip[k].uz/=norm; }
                    }
                    double e_new = single_energy(k);
                    if (!(e_new - e_old < 0 || randf() < exp(-(e_new-e_old)/(KB_RED*T_red)))) {
                        pos[k] = oldp; dip[k] = oldd;
                    }
                }
            }
            
            /* Production */
            double sum_E = 0, sum_E2 = 0, sum_align = 0;
            int nsamples = 0;
            for (int step = 0; step < nprod; step++) {
                for (int m = 0; m < N; m++) {
                    int k = rand() % N;
                    double e_old = single_energy(k);
                    Vec3 oldp = pos[k]; Dip oldd = dip[k];
                    total++;
                    if (randf() < 0.5) {
                        pos[k].x += randf2()*dr;
                        pos[k].y += randf2()*dr;
                        pos[k].z += randf2()*dr;
                    } else {
                        double th = dtheta*randf2(), ph = 2*M_PI*randf();
                        double ct=cos(th), st=sin(th), cp=cos(ph), sp=sin(ph);
                        double ux=dip[k].ux, uy=dip[k].uy, uz=dip[k].uz;
                        dip[k].ux = ux*ct + st*cp;
                        dip[k].uy = uy*ct + st*sp;
                        dip[k].uz = uz*ct;
                        double norm = sqrt(dip[k].ux*dip[k].ux+dip[k].uy*dip[k].uy+dip[k].uz*dip[k].uz);
                        if (norm > 0) { dip[k].ux/=norm; dip[k].uy/=norm; dip[k].uz/=norm; }
                    }
                    double e_new = single_energy(k);
                    if (e_new - e_old < 0 || randf() < exp(-(e_new-e_old)/(KB_RED*T_red))) {
                        accept++;
                    } else {
                        pos[k] = oldp; dip[k] = oldd;
                    }
                }
                if (step % 3 == 0) {
                    double etot = total_energy();
                    double align = 0;
                    for (int i = 0; i < N; i++) align += dip[i].uz;
                    align /= N;
                    sum_E += etot; sum_E2 += etot*etot;
                    sum_align += fabs(align);
                    nsamples++;
                }
            }
            double E_avg = sum_E / nsamples;
            double E_std = sqrt(fabs(sum_E2/nsamples - E_avg*E_avg));
            double align_avg = sum_align / nsamples;
            double acc_rate = (double)accept / total;
            printf("%.0f,%.0e,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   T_K, fields_SI[fi], N, E_avg, E_std, E_avg/N, align_avg, acc_rate);
            fflush(stdout);
        }
    }
    return 0;
}