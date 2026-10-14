#pragma once
#include "model.cuh"

// Host-side conversion of a symmetric COM inertia tensor. Jacobi rotations
// preserve a right-handed principal frame, including repeated eigenvalues.
struct PfInertiaTensor { float xx, yy, zz, xy, xz, yz; };
static inline bool pf_principal_inertia(PfInertiaTensor tensor,
        PfVec3& moments, PfQuat& rotation) {
    double a[3][3]={{tensor.xx,tensor.xy,tensor.xz},
        {tensor.xy,tensor.yy,tensor.yz},{tensor.xz,tensor.yz,tensor.zz}};
    double frame[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    double scale=0;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        if (!isfinite(a[i][j])) return false;
        scale=fmax(scale,fabs(a[i][j]));
    }
    if (scale==0) { moments=pf_v3(0,0,0); rotation=pf_quat_identity(); return true; }
    for (int sweep=0;sweep<32;++sweep) {
        int p=0,q=1;
        if (fabs(a[0][2])>fabs(a[p][q])) { p=0; q=2; }
        if (fabs(a[1][2])>fabs(a[p][q])) { p=1; q=2; }
        if (fabs(a[p][q])<=1.0e-14*scale) break;
        double angle=0.5*atan2(2*a[p][q],a[q][q]-a[p][p]);
        double c=cos(angle),s=sin(angle),ap=a[p][p],aq=a[q][q],off=a[p][q];
        a[p][p]=c*c*ap-2*c*s*off+s*s*aq;
        a[q][q]=s*s*ap+2*c*s*off+c*c*aq;
        a[p][q]=a[q][p]=0;
        for (int k=0;k<3;++k) {
            if (k!=p && k!=q) {
                double x=a[k][p],y=a[k][q];
                a[k][p]=a[p][k]=c*x-s*y;
                a[k][q]=a[q][k]=s*x+c*y;
            }
            double x=frame[k][p],y=frame[k][q];
            frame[k][p]=c*x-s*y; frame[k][q]=s*x+c*y;
        }
    }
    for (int i=0;i<3;++i) {
        if (a[i][i]<=0 || a[i][i]>a[(i+1)%3][(i+1)%3]+a[(i+2)%3][(i+2)%3]+1e-6*scale)
            return false;
        for (int j=0;j<i;++j) if (fabs(a[i][j])>1e-12*scale) return false;
    }
    double trace=frame[0][0]+frame[1][1]+frame[2][2];
    double v[4]; // wxyz
    if (trace>0) {
        double t=2*sqrt(trace+1);
        v[0]=0.25*t; v[1]=(frame[2][1]-frame[1][2])/t;
        v[2]=(frame[0][2]-frame[2][0])/t; v[3]=(frame[1][0]-frame[0][1])/t;
    } else {
        int i=0;
        if (frame[1][1]>frame[i][i]) i=1;
        if (frame[2][2]>frame[i][i]) i=2;
        int j=(i+1)%3,k=(i+2)%3;
        double t=2*sqrt(1+frame[i][i]-frame[j][j]-frame[k][k]);
        v[0]=(frame[k][j]-frame[j][k])/t; v[i+1]=0.25*t;
        v[j+1]=(frame[j][i]+frame[i][j])/t; v[k+1]=(frame[k][i]+frame[i][k])/t;
    }
    PfQuat orientation=pf_quat_normalize({(float)v[0],(float)v[1],(float)v[2],(float)v[3]});
    // Check reconstruction after conversion to float; reject instead of silently
    // dropping off-diagonal terms or returning an inaccurate principal frame.
    PfVec3 axes[3]; pf_quat_axes(orientation,axes);
    double original[3][3]={{tensor.xx,tensor.xy,tensor.xz},
        {tensor.xy,tensor.yy,tensor.yz},{tensor.xz,tensor.yz,tensor.zz}};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        double reconstructed=0;
        for (int k=0;k<3;++k) {
            float xyz[3]={axes[k].x,axes[k].y,axes[k].z};
            reconstructed+=xyz[i]*a[k][k]*xyz[j];
        }
        if (fabs(reconstructed-original[i][j])>2e-6*scale) return false;
    }
    moments=pf_v3((float)a[0][0],(float)a[1][1],(float)a[2][2]);
    rotation=orientation;
    return true;
}
