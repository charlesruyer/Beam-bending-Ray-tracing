# -----------------------------------------------------------------------
# Brillouin - C. Ruyer - 12/2019
# -----------------------------------------------------------------------
#
# >>>>>> Analyse de l'XP omega EP de traverse de cavite
#
# >>>>>> Requirements
#   python2.7 with the following packages: numpy, matplotlib, pylab, scipy

# >>>>>> Advice: install ipython (historics of commands, better interface)
# >>>>>> First step: invoke python and load this file
#      $ ipython -i Luke_xp.py
#
# >>>>>> Second step: in the ipython shell, use the functions

from scipy.linalg import expm, inv

import scipy.special
from scipy import signal
from scipy.special import exp1
from scipy.special import  fresnel
import numpy as np
import numpy.matlib as ma
from scipy.interpolate import interp1d
import os.path, glob, re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from scipy.integrate import odeint
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy.special import sici
from scipy.special import  lambertw
from scipy.special import  erf
from scipy.special import  erfi
from scipy import integrate
from scipy import optimize 
import pylab
import sys
#pylab.ion()

#Fonction plasma et ses derive a variable REELLE !
def Z_scalar(x):
    if np.abs(x)<=25:
        res = np.pi**0.5*np.exp(-x**2)*( 1j - erfi(x) ) 
    else:
        res=-1/x -1./x**3/2. +1j*np.sqrt(np.pi)*np.exp(-x**2)
    return res
def Z(x):
    if np.size(x)==1:
        return Z_scalar(x)
    else:
        res = 0*x+0j*x
        for i in range(len(x)):
            ##print 'i= ',i, x[i], Z_scalar(x[i])
            res[i] = Z_scalar(x[i]) 
        return res 

def Zp(x):
    return -2*(1 + x*Z(x)) 

def Zp_scalar(x):
    return -2*(1 + x*Z_scalar(x))

def Zpp(x):
    return  -2*(Z(x) + x*Zp(x))

def Zppp(x):
    return  -2*( 2*Zp(x) + x*Zpp(x) )

def nuIAW(Z,A,Te,Ti):
    c=3e8
    cs=np.sqrt((Z*Te+3*Ti)/(1836.*A*511000.))*c
    xi=cs*np.sqrt(511000.*1836.*A/(2*Ti))/c
    xe=cs*np.sqrt(511000./(2*Te))/c
    Zpi,Zpe = Zp(xi), Zp(xe)
    Zppi, Zppe = Zpp(xi), Zpp(xe)
    g0_exact = +1/cs* np.imag(  (Zpi+Ti/(Z*Te)*Zpe)/(Zppi*np.sqrt(511000*1836*A/(2*Ti))/c +Ti/(Z*Te)*Zppe*np.sqrt(511000/(2*Te))/c  )   )
    vi=np.sqrt(Ti/(1836*511000.*A))*c
    g01 = np.sqrt(np.pi/8)*(cs/vi)**3*(np.exp(-xi**2) +np.sqrt(Z/(A*1836.))*np.sqrt(Ti/(Z*Te))**3 )
    
    print('g0 exact = ',g0_exact)
    print('g0 xi>>1, xe<<1 = ',g01)

def plot_alpha_kin(figure=1,Te=1.e3, ksk0max=0.02,k0=2*np.pi/0.35e-6,Z=[1.], A=[1.], nisne=[1.],nesnc=0.1):
    c=3e8
    #Ti=1000.
    Z=np.array(Z)
    vec=Z/Z
    #cs = 0.5*np.sqrt((Te+3*Ti)/511000./1836.)*c
    #vd=0.8*cs
    w0=k0*c/np.sqrt(1-nesnc)
    wpe2 = w0**2*nesnc
    ks = np.linspace(-ksk0max*k0,ksk0max*k0,1000)
    vp = (-0.5*np.abs(ks) *c**2/w0 -0*ks/np.abs(ks))
    #ak1 = alpha_kin(xie,xii,1)
    lde2   = Te/511000.*c / wpe2
    k2lde2 = ks**2 * lde2
    ak1 = alpha_kin(Te=Te, Ti=Te/1.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 
    ak3 = alpha_kin(Te=Te, Ti=Te/3.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 
    ak5 = alpha_kin(Te=Te, Ti=Te/5.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 

    ak1d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/1.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)
    ak3d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/3.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)
    ak5d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/5.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)

    #print ak1d2
    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        l1,=plt.plot(ks/k0, np.real(ak1), 'k',linewidth=2 )
        l3,=plt.plot(ks/k0, np.real(ak3) , 'r',linewidth=2)
        l5,=plt.plot(ks/k0, np.real(ak5), 'b',linewidth=2 )
        l1d,=plt.plot(ks/k0, np.real(ak1d2), '--k',linewidth=2 )
        l3d,=plt.plot(ks/k0, np.real(ak3d2) , '--r',linewidth=2)
        l5d,=plt.plot(ks/k0, np.real(ak5d2), '--b',linewidth=2 )
        prop = fm.FontProperties(size=18)
        plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel("$k_s/k_0$")
        ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        if ksk0max<0.025:
            ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        ax.set_xlim(-ksk0max,ksk0max)
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        plt.show()

        # plt.rcParams.update({'font.size': 20})
        # fig = plt.figure(figure,figsize=[7,5])
        # fig.clf()
        # ax = fig.add_subplot(1,1,1)
        # plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        # l1,=plt.plot(ks/k0, np.imag(ak1), 'k',linewidth=2 )
        # l3,=plt.plot(ks/k0, np.imag(ak3) , 'r',linewidth=2)
        # l5,=plt.plot(ks/k0, np.imag(ak5), 'b',linewidth=2 )
        # l1d,=plt.plot(ks/k0, np.imag(ak1d2), '--k',linewidth=2 )
        # l3d,=plt.plot(ks/k0, np.imag(ak3d2) , '--r',linewidth=2)
        # l5d,=plt.plot(ks/k0, np.imag(ak5d2), '--b',linewidth=2 )
        # prop = fm.FontProperties(size=18)
        # plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        # ax.set_xlabel("$k_s/k_0$")
        # ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        # if ksk0max<0.025:
        #     ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        # ax.set_xlim(-ksk0max,ksk0max)
        # #ax.set_xscale("log")
        # #ax.set_yscale("log")
        # plt.show()


# Fonction de transfert cinetique de la force ponderomotrice
# Zp(xie)/2 * ( 1 - sum_i Zp(xi) ) /eps
def alpha_kin(Te, Ti, Z, A, nisne, vphi, k2lde2=0, ne=1,figure=None, is_chiperp = False):
    c=3e8
    if ne ==0 : # On neglige les e-
        Zpxie = -2+0j
    else: 
        xie = vphi /np.sqrt(2*Te/511000. )/c
        Zpxie = Zp(xie)
    ##print 'Zp(xie) = ', Zpxie
    Xe= Zpxie * 1.
    sumXi = 0. + 0j
    for i in range(len(Ti)):
        xi = vphi /np.sqrt(2*Ti[i]/511000./1836/A[i] )/c
        sumXi += Zp(xi) * nisne[i] *Te* Z[i]**2/Ti[i]
        
        ##print 'Zp(xii) = ', Zp(xi)
    # Si k2lde2 ==0  on suppose k2lde2 <<  1 (Longeur de Debaye electronique
    ak = -0.5*Zpxie *(k2lde2- sumXi) / (k2lde2 - Xe-sumXi)
    ##print np.shape(np.real(ak)), np.shape(np.imag(ak))

    fluctuation=np.imag(1./ (+Xe+sumXi))
    if figure is None:
        return  ak
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,np.real(ak),'k', linewidth=2)              
        a, = ax.plot(vphi/c,np.imag(ak),'--k', linewidth=2)              
        ax.set_ylabel("$F_\mathrm{kin}$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,fluctuation,'k', linewidth=2)                
        ax.set_ylabel("$\\Im( 1./\\epsilon)$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()




def taux_collisionnel(masse=None,charge=None,dens=None,temp=None,vd=None,*aqrgs,**kwargs):
    #% Frequences de collision de Spitzer
    #% Ref. A. Decoster, Modeling of collisions (1998)
    #%
    #% Input : [beam, background]
    #% . masse(1:2)  : masse/me
    #% . charge(1:2) : charge/qe
    #% . dens(1:2)   : densite (cm^-3)
    #% . temp(1:2)   : temperature (eV)
    #% . vde(1:2)    : vitesses de derive (cm/s)
    #% Output :
    #% . nu_imp  : momentum transfer frequency (1/s)
    #% . nu_ener : Energy transfer frequency (1/s)
    #% . lambda  : mean-free-path (cm)
    masse = np.array(masse)
    charge = np.array(charge)
    dens = np.array(dens)
    temp = np.array(temp)
    vd = np.array(vd)
    
    #varargin = cellarray(args)
    #nargin = 5-[masse,charge,dens,temp,vd].count(None)+len(args)
    
    vt1=30000000000.0 * (temp[0] / (masse[0] * 511000.0)) ** 0.5
    vd1=np.abs(vd[0])
    if all(masse == 1):
        ##print 'All electrons'
        if temp[1] < 10:
            log_coul=23 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1.5))
        else:
            log_coul=24 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1))
        ##print log_coul, temp
    else:
        if any(masse == 1.):
            ##print 'electron-ion'
            if masse[0] ==1.:
                ielec=0
                ##print 'indice electron: ',ielec
                iion=1
                ##print 'indice ion: ',iion
            else:
                ielec=1
                ##print 'indice electron: ',ielec
                iion=0
                ##print 'indice ion: ',iion
            
            
            if (temp[iion] / masse[iion] < temp[ielec]) and (temp[ielec] < 10 * charge[iion] ** 2):
                log_coul=23 - np.log(dens[ielec] ** 0.5 * charge[iion] * temp[ielec] ** (- 1.5))
            else:
                if 10 * charge[iion] ** 2 < temp[ielec]:
                    log_coul=24 - np.log(dens[ielec] ** 0.5 * temp[ielec] ** (- 1))
                else:
                    if temp[ielec] < temp[iion] * charge[iion] / masse[iion]:
                        mu=masse[iion]/1836.
                        log_coul=30 - np.log(dens[iion] ** 0.5 * temp[iion] **(-1.5) * charge[iion] ** 2 / mu)
                    else:
                        print( 'No Coulombien logarithm from Lee and Moore')
                        return  {"nup":None,"nuk":None,"log_coul":None,"lmfp":None}
            	##print 'Log Coulombien: ',log_coul
        else:
            log_coul=23 - np.log(charge[0] * charge[1] * (masse[0] + masse[1]) * (dens[0] * charge[0] ** 2 / temp[0] + dens[1] * charge[1] ** 2 / temp[1]) ** 0.5 / (masse[0] * temp[1] + masse[1] * temp[0]))

    qe=4.8032e-10
    temp=1.6022e-19 * 10000000.0 * temp
    masse=9.1094e-28 * masse
    m12=masse[0] * masse[1] / (masse[0] + masse[1])
    nu_imp=(4. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * m12 * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)
    nu_ener=(8. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * masse[1] * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)

    _lambda=np.max([vt1,vd1]) / nu_imp
    ##print 'nu_imp = ',nu_imp,' Hz'
    ##print 'nu_ener = ',nu_ener,' Hz'
    ##print 'tau_imp = ',1/nu_imp,' s'
    ##print 'tau_ener = ',1/nu_ener,' s', dens, log_coul,masse
    ##print 'log_coul = ',log_coul
    ##print 'Mean-free-path: ',_lambda,' cm'
    result = {"nup":nu_imp,"nuk":nu_ener,"log_coul":log_coul,"lmfp":_lambda}
    return result


def dispe_filam_kin(Te=1.e3,Ti=[300.],Z=[1.],A=[1.],nesnc=0.1,I0=3e14,k0=2*np.pi/0.35e-6,nisne=None,figure=1):
    c=3e8
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.array(Z))]
        print( 'nisne = ',nisne)
    dnsn = I0*1e4/(nc*c*Te*1.6e-19)
    print('dnsn = ',dnsn)
    n=np.sqrt(1-nesnc)
    cs = np.sqrt((Z[0]*Te+3*Ti[0])/(1836.*511000.*A[0] ))*c
    k = np.linspace(0,2*k0*dnsn**0.5,120)
    m=0
    for i in range(len(Ti)):
        m+=Z[i]**2*nisne[i]*Te/Ti[i]
    alphak = m/(1.+m)
    uf2 =0.25 *( 0.5*dnsn/n**2*alphak - k**2/k0**2)
    ik = uf2>0

    kf,uf =k[ik], np.sqrt(uf2[ik])
    u2 = 1+ 0.5/uf2 *(1.-np.sqrt(1+4*uf2))
    ik=u2>0
    ku,u = k[ik],np.sqrt(u2[ik])

    ne=nesnc*nc/1e6
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])
    print( r['lmfp'])
    lmfp=r['lmfp']*1e-2

    print( 'lmfp = ',lmfp,' m')
    x=lmfp*Z[0]**0.5*k
    Ak =2*( 0.5 +Z[0]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )
    ufa2 = 0.25*( 0.5*dnsn/n**2*alphak*Ak - k**2/k0**2 )
    ik = ufa2>0
    kfa,ufa =k[ik], np.sqrt(ufa2[ik])
    ua2 = 1+ 0.5/ufa2 *(1.-np.sqrt(1+4*ufa2))
    ik=ua2>0
    kua,ua = k[ik],np.sqrt(ua2[ik])
    #Brillouin avant 

    k=np.linspace(0,1.1*np.max(kua),120)
    x=lmfp*Z[0]**0.5*k
    Ak =2*( 0.5 +Z[0]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )

    vphi = np.linspace(0.*cs, 1.2*cs,100)
    V, K=np.meshgrid(vphi,k)
    alphakold=alphak
    alphak = alpha_kin(Te,Ti, Z, A, nisne, (vphi), k2lde2=0, ne=1)
    print( np.shape(alphak), np.shape(Ak))
    U2 = 0*V +0j*V
    U = 0*V +0j*V
    for iv in range(len(vphi)):
        for ik in range(len(k)):
            #U2[ik,iv] =0.25*( 0.5*dnsn/n**2*alphak[iv]*Ak[ik] - k[ik]**2/k0**2 )
            b=2j*vphi[iv]/n/c
            UF2= 0.25*( 0.5*dnsn/n**2*alphak[iv]*Ak[ik] - k[ik]**2/k0**2 )
            delta = b**2-4*(-UF2-vphi[iv]**2/n**2/c**2)
            thd=np.arctan(np.real(delta),np.imag(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            U[ik,iv] = 0.5*(-b+rd**0.5*np.exp(1j*thd/2.)) 
            #ur,uth = np.abs(UF2), np.arctan(np.real(UF2),np.imag(UF2))+np.pi*(np.real(UF2)<0)
            #U[ik,iv] = ur**0.5*np.exp(1j*uth/2.)
    gamma = K*np.real(U)
    kpara = -K*np.imag(U)
    #np.arctan2(np.real(U2),np.imag(U2))/2.
    #uabs = np.sqrt(np.abs(U2))
    #gamma = K*uabs*np.cos(uth)
    #kpara = -K*uabs*np.sin(uth)

    #mesh  = gamma>0
    #gamma *= mesh
    #kpara *= mesh
    for iv in range(len(vphi)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,iv]) or np.isnan(kpara[ik,iv]):
                gamma[ik,:iv] = 0
                kpara[ik,:iv] = 0


    print( 'gamma/k0 = ',gamma /k0)
    print( 'kpara/k0 = ',kpara/k0)
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)
        plt.plot(ku/k0 , ku*u/k0,'k',linewidth=2)
        plt.plot(kf/k0 , kf*uf/k0,'--k',linewidth=2)
        plt.plot(kua/k0 , kua*ua/k0,'r',linewidth=2)
        plt.plot(kfa/k0 , kfa*ufa/k0,'--r',linewidth=2)

        plt.plot(k/k0,np.squeeze(gamma[:,0])/k0,'g',linewidth=2)
        ax0.set_ylabel("$\\Gamma/k_0$")
        ax0.set_xlabel("$k_\\perp/k_0$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)#, vmin =0, vmax =1.5*np.max(kfa*ufa/k0) )
        plt.colorbar(cf0)#, ticks=ct)
        ax0.set_xlabel("$k/k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)
        data=kpara.T/gamma.T
        #data = np.log10(gamma)
        cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)#, vmin =np.max(data)-2, vmax =np.max(data) )
        plt.colorbar(cf0)#, ticks=ct)
        ax0.set_xlabel("$k/k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

def plot_bb_ssd_param2ax(M,taussd,nesnc=0.1,I0=3e14,Te=1e3,nc=1e27,ztesti=1.,fnum=6.5,Z=[1.],A=[1.],nisne=[1.],figure=1 ):
    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    cs = np.sqrt((Te+3*Te/ztesti)/511000./1836.)*c
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print( 'l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    MM,Tau = np.meshgrid(M,taussd)
    Dth=0*MM
    dnsn = I0*1e4/(Te*1.6e-19*nc*c*n)
    print('dnsn = ',dnsn)

    #regime transitoire
    vte = np.sqrt(Te/511000.)
    g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-A[0]*1836*9.1e-31*cs**2/(2*Te/ztesti*1.6e-19))/(2*Te/ztesti/511000.)**1.5 +A[0]*1836./Z[0]/(2*Te/511000.)**1.5)
    print( 'g_0 = ', g0)
    w=2*cs/sigma/np.sqrt(np.pi)
    nu=g0*w
    print( '1/nu = ',1./nu)
    a=np.sqrt(w**2-nu**2)
    def f(t):
        return 1-np.exp(-nu*t)*( (np.cos(a*t)+nu/a*np.sin(a*t))*np.cos(w*t) +w/a*np.sin(a*t)*np.sin(w*t) )

    #abondance de speckle
    u=np.linspace(2.,10,10000)
    g3 = np.pi**1.5*5**0.5/27.*(u**1.5-3./10.*u**0.5)*np.exp(-u)
    g2 = np.pi/3./np.sqrt(15.)*((0.5+np.pi/4.)*u +0.5)*np.exp(-u)
    fu = g2/(sigma*zc)
    fu = fu/np.trapz(fu,u)
    fumoy = np.trapz(u*fu,u)/np.trapz(fu,u)
    fu2moy = np.sqrt(np.trapz(u**2*fu,u)/np.trapz(fu,u))
    print('<I>/I_0 = ',fumoy)
    print( '$<I^2>^{1/2}/I_0$ = ',fumoy)
    
    vp = -M*cs
    akin=-np.imag(alpha_kin(Te=Te, Ti=Te/ztesti*vec,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
   
    fmoy = np.mean(f(np.linspace(0,taussd,1000)))
    Dthu = fmoy*0.5*nesnc*dnsn*akin*zc/sigma/np.sqrt(np.pi) *u

    Dthm = 1.#np.pi/180.

    if figure is None:
        return Dth.T
    else:
        cscale='lin'
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        ax1=plt.twinx()
        plt.subplots_adjust(left=0.22, right=0.84, top=0.9, bottom=0.2)
        ax0.plot(u, Dthu*1e3, 'k', linewidth=2)
        ax1.plot(u, fu, 'r', linewidth=2)
        ax0.set_ylabel("$\\Delta \\theta$ ($\\mu$m /mm)")
        ax1.set_ylabel("f(I)", color='red')
        ax1.set_yscale("log")
        ax0.set_xlabel("$I/I_0$")
        #ax0.set_xlim(np.min(taussd*1e12),np.max(taussd*1e12))
        fig.canvas.draw()
        plt.show()
        

def plot_bb_ssd_param(Te,taussd,M=0.9,nesnc=0.2,I0=5*3e14,nc=9e27,ztesti=1.,fnum=8.,Z=[1.],A=[1.],nisne=[1.],figure=1 ):
    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    
    cs = np.sqrt((Te+3*Te/ztesti)/511000./1836.)*c
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print( 'l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    TE,Tau = np.meshgrid(Te,taussd)
    
    CS = np.sqrt((TE+3*TE/ztesti)/511000./1836.)*c
    Dth=0*TE
    Dthmax=0*TE
    dnsn = I0*1e4/(TE*1.6e-19*nc*c*n)
    #print 'dnsn = ',dnsn

    #regime transitoire
    vte = np.sqrt(TE/511000.)
    #g0= np.sqrt(np.pi) *(CS/c)**3*(np.exp(-A[0]*1836*9.1e-31*CS**2/(2*TE/ztesti*1.6e-19))/(2*TE/ztesti/511000.)**1.5 +A[0]*1836./Z[0]/(2*TE/511000.)**1.5)
    vi=np.sqrt(TE*Z[0]/ztesti/(511000.*1836*A[0]))*c
    g0= np.sqrt(np.pi/8) *(CS/vi)**3*(np.exp(-0.5*(CS/vi)**2 ) +np.sqrt(Z[0]/(A[0]*1836.)) *(ztesti)**-1.5)

    #print 'g_0 = ', g0
    kc=2**0.5/sigma
    w=kc*CS 
    nu=g0*w
    #print '1/nu = ',1./nu
    a=np.sqrt(w**2-nu**2)
    
    def f(t,nuu,ww,aa,vv):
        return 1-np.exp(-nuu*t)*( (np.cos(aa*t)+nuu/aa*np.sin(aa*t))*np.cos(kc*vv*t)  )

    
    theta = np.linspace(0,np.pi/2,5000)
    for it in range(len(Te)):
        vp = -M*cs[it]*np.cos(theta)
        akin=-np.imag(alpha_kin(Te=Te[it], Ti=Te[it]/ztesti*vec,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
        beta = np.trapz(akin*np.cos(theta),theta)/2**1.5
        
        #Calcul non local
        ne=nesnc*nc/1e6
        #print [1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te,Ti[0]]
        r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te[it],Te[it]/ztesti],vd=[0,0])
        #print r['lmfp']
        lmfp=r['lmfp']*1e-2
        #print 'lmfp = ',lmfp,' m'
        x=2*lmfp*Z[0]**0.5/(sigma *np.pi**0.5)
        Ak =2*( 0.5 +Z[0]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )
        print( "it = ",it," / ",len(Te),", Non local, Ak = ", Ak)
        for iss in range(len(taussd)):
            fmoy = np.mean(f(np.linspace(0,taussd[iss],1000),nu[iss,it],w[iss,it],a[iss,it],M*cs[it]))
            fmax = f(taussd[iss],nu[iss,it],w[iss,it],a[iss,it],M*cs[it] )
            Dth[iss,it] = Ak*fmoy*0.5*nesnc*dnsn[iss,it]*beta*zc/sigma/np.sqrt(np.pi)
            Dthmax[iss,it] = Ak*fmax*0.5*nesnc*dnsn[iss,it]*beta*zc/sigma/np.sqrt(np.pi)
    Dthm = 1.#np.pi/180.

    if figure is None:
        return Dth.T
    else:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.935, top=0.9, bottom=0.2)
        data=Dth.T
        data = np.log10(data)
        cf0 = plt.pcolor(taussd*1e12,Te,data,cmap=plt.cm.gist_earth_r, vmin =np.max(data)-3, vmax =np.max(data) )
        plt.colorbar(cf0, ticks=[-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5])
        import matplotlib
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        CS1=ax0.contour(taussd*1e12,Te,np.log10(Dth.T),(-3,-2,-1) ,colors='k')
        
        ax0.clabel(CS1, fontsize=15, inline=1)
        CS2=ax0.contour(taussd*1e12,Te,np.log10(Dthmax.T), (-3,-2,-1),colors='r')
        b,=plt.plot([],[],'k')
        r,=plt.plot([],[],'r')
        prop = fm.FontProperties(size=20)
        ax0.legend([b,r],
                   ['$\log_{10}\langle\Delta \\theta \\rangle_\mathrm{SSd}$', 
                    '$\log_{10}\Delta \\theta (\\tau_\mathrm{SSd})$'],
                   loc=1,prop=prop,
                   bbox_to_anchor=(1,1.15))
        
        ax0.clabel(CS2, fontsize=15, inline=1)

        ax0.set_ylabel("$T_e$ (eV)")
        ax0.set_xlabel("$\\tau_\\mathrm{SSD}$ (ps)")
        ax0.set_xlim(np.min(taussd*1e12),np.max(taussd*1e12))
        ax0.set_ylim(np.min(Te),np.max(Te))
        if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
            ax0.set_xscale('log')
        if Te[1]-Te[0] !=Te[2]-Te[1]:
            ax0.set_yscale('log')
        fig.canvas.draw()
        plt.show()
        
# =============================================================================
#         plt.rcParams.update({'font.size': 25})
#         fig = plt.figure(figure+1,figsize=[7,6])
#         fig.clf()
#         ax0 = fig.add_subplot(1,1,1)
#         plt.subplots_adjust(left=0.18, right=0.935, top=0.9, bottom=0.2)
#         data=Dthmax.T
#         data = np.log10(data)
#         cf0 = plt.pcolor(taussd*1e12,Te,data,cmap=plt.cm.gist_earth_r, vmin =np.max(data)-3, vmax =np.max(data) )
#         plt.colorbar(cf0, ticks=[-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0])
#         ax0.set_ylabel("$T_e$ (eV)")
#         ax0.set_xlabel("$\\tau_\\mathrm{SSD}$ (ps)")
#         ax0.set_xlim(np.min(taussd*1e12),np.max(taussd*1e12))
#         ax0.set_ylim(np.min(Te),np.max(Te))
#         if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
#             ax0.set_xscale('log')
#         if Te[1]-Te[0] !=Te[2]-Te[1]:
#             ax0.set_yscale('log')
#         fig.canvas.draw()
#         plt.show()
# =============================================================================
        
def plot_bb_param(M,ztesti,nesnc=0.1,I0=3e14,Te=1e3,nc=1e27,fnum=6.5,Z=[1.],A=[1.],nisne=[1.],figure=1 ):
    #Aritcle beam bending part 1, Fig 3:
    # (a) plot_bb_param(M=np.linspace(0,1.8,50),ztesti=np.linspace(0.1,10,50),nesnc=0.2,fnum=8,nc=9e27,I0=3e15,Te=2e3)
    # (b) plot_bb_param(M=np.linspace(0,1.8,50),ztesti=np.linspace(0.1,10,50),nesnc=0.2,fnum=8,nc=9e27,I0=3e15,Te=2e3,Z=[1.,6.],A=[1.,12.],nisne=[1./7.,1./7.])

    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    
    Zm,Am=np.mean(np.array(Z)), np.mean(np.array(A))
    cs = np.sqrt((1*Te+3*Te/ztesti)/511000./1836./1)*c
    zeta= Zm*Te/511000./(1836.*Am*cs**2/c**2)
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print( 'l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    MM,Ztesti = np.meshgrid(M,ztesti)
    Dth=0*MM
    Dthf =0*MM
    dnsn = I0*1e4/(Te*1.6e-19*nc*c*n)
    print ('dnsn = ',dnsn)

    #abondance de speckle
    u=np.linspace(2.,10,10000)
    g3 = np.pi**1.5*5**0.5/27.*(u**1.5-3./10.*u**0.5)*np.exp(-u)
    g2 = np.pi/3./np.sqrt(15.)*((0.5+np.pi/4.)*u +0.5)*np.exp(-u)
    fu = g2/(sigma*zc)
    fu = fu/np.trapz(fu,u)
    fumoy = np.trapz(u*fu,u)/np.trapz(fu,u)
    fu2moy = np.sqrt(np.trapz(u**2*fu,u)/np.trapz(fu,u))
    print( '<I>/I_0 = ',fumoy)
    print( '$<I^2>^{1/2}/I_0$ = ',fumoy)
    
    th=np.linspace(0,np.pi/2,2000)
    for im in range(len(M)):
        for iz in range(len(ztesti)):
            vp = -M[im]*cs[iz]*np.cos(th)
            akin=-np.imag(alpha_kin(Te=Te, Ti=Te/ztesti[iz]*vec,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
            Ti=Zm*Te/ztesti[iz]
            vi = np.sqrt(Ti/511000./1836./Am)*c
            g0= np.sqrt(np.pi/8) *(cs[iz]/vi)**3*(np.exp(-0.5*(cs[iz]/vi)**2 ) +np.sqrt(Zm/(Am*1836.)) *(Ti/Te/Zm)**1.5)
            #g0= np.sqrt(np.pi) *(cs[iz]/c)**3*(np.exp(-Am*1836*9.1e-31*cs[iz]**2/(2*Ti*1.6e-19))/(2*Ti/511000.)**1.5 +Am*1836./Zm/(2*Te/511000.)**1.5)

            af= zeta[iz]*np.imag(cs[iz]**2/(cs[iz]**2-vp**2 + 2*1j*g0 *cs[iz]*vp ))
            b3d=np.trapz(akin*np.cos(th),th)/2**0.5
            b3df=np.trapz(af*np.cos(th),th)/2**0.5
            
            Dth[iz,im] = 1*0.5*nesnc*dnsn*b3d*zc/sigma/np.sqrt(np.pi)
            Dthf[iz,im] = 1*0.5*nesnc*dnsn*b3df*zc/sigma/np.sqrt(np.pi)
    Dthm = 1.#np.pi/180.

    if figure is None:
        return Dth.T
    else:
        
        vma =np.max([np.max(np.log10(Dth)),np.max(np.log10(Dthf))])
        vmi = vma-2
        
        vma =np.max([np.max((Dth)),np.max((Dthf))])
        vmi = 0
        
        cscale='lin'
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.9, top=0.9, bottom=0.2)
        data=Dth.T
        #data = np.log10(data)
        cf0 = plt.pcolor(ztesti,M,data,cmap=plt.cm.gist_earth_r, vmin =vmi, vmax =vma )
        plt.colorbar(cf0)#,ticks=[-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0])#, ticks=ct)
        ax0.set_ylabel("$M_0$")
        ax0.set_xlabel("$T_e/T_i$")
        ax0.set_xticks([2,4,6,8,10,12])
        ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
  
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.9, top=0.9, bottom=0.2)
        data=Dthf.T
        #data = np.log10(data)
        cf0 = plt.pcolor(ztesti,M,data,cmap=plt.cm.gist_earth_r,vmin =vmi, vmax =vma)
        plt.colorbar(cf0)#,ticks=[-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0])#, ticks=ct)
        ax0.set_ylabel("$M_0$")
        ax0.set_xlabel("$T_e/T_i$")
        ax0.set_xticks([2,4,6,8,10,12])
        ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
def bb_3d(t,L,M,ztesti,nesnc=0.1,I0=3e14,Te=1e3,nc=1e27,fnum=6.5,Z=[1.],A=[1.],nisne=[1.],figure=1 ):

    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    cs = np.sqrt((Te+3*Te/ztesti)/511000./1836.)*c
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print( 'l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    dnsn = I0*1e4/(Te*1.6e-19*nc*c*n)
    print('dnsn = ',dnsn)
    
    th=np.linspace(0,np.pi/2,2000)
    
    vp = -M*cs*np.cos(th)
    akin=-np.imag(alpha_kin(Te=Te, Ti=Te/ztesti*vec,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
    b3d=np.trapz(akin*np.cos(th),th)/2**0.5
    Dth = 1*0.5*nesnc*dnsn*b3d*L/sigma/np.sqrt(np.pi) /np.sqrt( 1 +(L/(2*zc))**2 )
    Dy = 1*0.5*nesnc*dnsn*b3d*L/sigma/np.sqrt(np.pi) /np.sqrt( 1 +(L/(2*zc))**2 )*L/2

    Ti=Te/ztesti
    vi = np.sqrt(Ti/(1*1836.*511000.))*3e8
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Z[0]/(A[0]*1836.)) *(Ti/Te/Z[0])**1.5)
    w=2*cs/sigma/np.sqrt(np.pi)
    nu=g0*w
    print( '1/nu = ',1./nu)
    #print '1/nu = ',1./nu
    a=np.sqrt(w**2-nu**2)
    def f(t,nuu,ww,aa):
        return 1-np.exp(-nuu*t)*( (np.cos(aa*t)+nuu/aa*np.sin(aa*t))*np.cos(ww*t) +ww/aa*np.sin(aa*t)*np.sin(ww*t) )
    

    print( 'Dtheta t=infinity = ',Dth)
    print( 'Dy =  t=infinity', Dy *1e6, ' microns')

    f=f(t,nu,w,a)
    print ('f(t) = ',f)

    print ('Dtheta  = ',Dth*f)
    print( 'Dy = ', Dy*f *1e6, ' microns')

    
def plot3d2d(M,ztesti=1.,nesnc=0.1,I0=6e14,Te=1e3,nc=1e27,fnum=6.5,Z=[1.],A=[1.],nisne=[1.],figure=1):
    #initilisation
    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    Ti= Te/ztesti*vec
    cs = np.sqrt((Te+3*Te/ztesti)/511000./1836.)*c
    zeta = Te/511000./(1836.* (cs/c)**2)
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print('l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    dnsn = I0*1e4/(Te*1.6e-19*nc*c*n)
    print( 'dnsn = ',dnsn)
    
    #Calcul non local
    ne=nesnc*nc/1e6
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])

    lmfp=r['lmfp']*1e-2
    print( 'lmfp = ',lmfp,' m')
    x=2*lmfp*Z[0]**0.5/(sigma *np.pi**0.5)
    Ak =2*( 0.5 +Z[0]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )
    print( "Non local, Ak = ", Ak)
    
                
    g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-A[0]*1836*9.1e-31*cs**2/(2*Ti[0]*1.6e-19))/(2*Ti[0]/511000.)**1.5 +A[0]*1836./Z[0]/(2*Te/511000.)**1.5)

    #Deflection 2D
    vp = -M*cs
    beta2d=-np.imag(alpha_kin(Te=Te, Ti=Ti,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
    Dth2d = beta2d    
    Dth2df =np.imag( zeta*cs**2/(cs**2-vp**2 +2j*vp*cs*g0) )
    Dth2dnl = Ak*beta2d  

    #Deflection 3d
    beta3d=0*M
    beta3df=0*M
    theta=np.linspace(0,np.pi/2.,5000)
    norma = nesnc*dnsn
    
    for im in range(len(M)):
        vp = -M[im]*cs*np.cos(theta)
        a3d = -np.imag(alpha_kin(Te=Te, Ti=Ti,Z=Z,A=A,nisne=nisne, vphi=vp,figure=None,k2lde2=0) )
        beta3d[im] =np.trapz(a3d*np.cos(theta),theta)/2**0.5
        beta3df[im]=np.trapz(np.imag( cs**2/(cs**2-vp**2 +2j*vp*cs*g0) )*np.cos(theta),theta)/2**0.5
    Dth3df =beta3df
    Dth3d = beta3d    
    Dth3dnl = Ak*beta3d   
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.195, right=0.95, top=0.9, bottom=0.2)
        t2d, = ax0.plot(M,Dth2d,'--k',linewidth=2 )
        #t2df, = ax0.plot(M,Dth2df,'--b',linewidth=2 )
        t2dnl, = ax0.plot(M,Dth2dnl,'--r',linewidth=2 )
        t3d, = ax0.plot(M,Dth3d,'k',linewidth=2 )
        #t3df, = ax0.plot(M,Dth3df,'b',linewidth=2 )
        t3dnl, = ax0.plot(M,Dth3dnl,'r',linewidth=2 )
        prop = fm.FontProperties(size=18)
        first_legend = ax0.legend([t2d,t3d],['Two dimension','Three dimension'],loc=8,prop=prop)#,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend([t3d,t3dnl],['Collisionless', 'Non local correction'],loc=1, prop=prop,bbox_to_anchor=(1.05, 1.14))
        
        ax0.set_xlabel("$M_0$")
        ax0.set_ylabel("$A_k\\beta_\mathrm{kin}$")
        ax0.set_xlim(np.min(M),np.max(M))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        
#Fig 2 du premier article du beam bending
def theory2d_article(figure=1):
    # %% Initialisation
    #initilisation
    nesnc=0.1
    Lx=114e-6
    c=3.e8
    n=np.sqrt(1-nesnc)
    l0=1e-6
    k0=2*np.pi/l0
    fnum=5.
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    M=np.linspace(0,1.5,100)
    
    ###########
    # %% Figure a
    ###########
    Z,A = 6.,12.
    Te=500.
    Ti= Z*Te 
    cs = np.sqrt((Z*Te+3*Ti)/511000./1836./A)*c
    dnsn=0.031
    vp = -M*cs
    H2,H3 = 1./(1+Lx**2/(4*zc**2))**0.5 , np.arctan(Lx/(2*zc))/ (Lx/(2*zc))
    #cinetique
    beta2d=-np.imag(alpha_kin(Te=Te, Ti=[Ti],Z=[Z],A=[A],nisne=[1./Z], vphi=vp,figure=None,k2lde2=0) )
    Dyk2d= nesnc*dnsn*beta2d *np.sqrt(2./np.pi)* Lx/(2*sigma) *H2 * Lx

    beta3d=0*M
    theta=np.linspace(0,np.pi/2.,10000)
    for im in range(len(M)):
        vp = -M[im]*cs*np.cos(theta)
        #beta3df[im]=np.trapz(np.imag( cs**2/(cs**2-vp**2 +2j*vp*cs*g0) )*np.cos(theta),theta)/2**0.5
        a3d = -np.imag(alpha_kin(Te=Te, Ti=[Ti],Z=[Z],A=[A],nisne=[1./Z], vphi=vp,figure=None,k2lde2=0) )
        beta3d[im] =np.trapz(a3d*np.cos(theta),theta)/2**0.5
    #Dyf2d= 1.36*nesnc*2*dnsn*0.5*beta2df *(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    #Dyf3d= 1.36*nesnc*2*dnsn*0.5*beta3df *(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    #Dyf2d= nesnc*dnsn*beta2df*np.sqrt(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    #Dyf3d= nesnc*dnsn*beta3df*np.sqrt(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    Dyk3d= nesnc*dnsn*beta3d*np.sqrt(2./np.pi)* Lx/(2*sigma) *H3*Lx

    #PIC
    Mb =np.array([0.5, 0.95,1.2,1.5])/2**0.5
    Dyb=np.array([0.7, 1.2,1.45,1.1])
    Mr =np.array([0.5, 0.9,1.2,1.5])/2**0.5
    Dyr=np.array([0.75, 1.3,1.5,1.3])
    
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        t2d, = ax0.plot(M,Dyk2d*1e6,'k',linewidth=2 )
        t3d, = ax0.plot(M,Dyk3d*1e6,'--k',linewidth=2 )
        #t2df,= ax0.plot(M,Dyf2d*1e6,'--k',linewidth=2 )
        #t3df,= ax0.plot(M,Dyf3d*1e6,'--c',linewidth=2 )
        r,   = ax0.plot(Mr,Dyr, 'or',ms=10)
        b,   = ax0.plot(Mb,Dyb, 'vb',ms=10)
        prop = fm.FontProperties(size=20)
        #first_legend = ax0.legend([t2d,t2df,t3df],['2D, kinetic','2D fluid','3D fluid'],loc=8,prop=prop)#,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        #plt.legend([t3d,t3dnl],['Collisionless', 'Non local correction'],loc=1, prop=prop,bbox_to_anchor=(1.05, 1.14))
        first_legend = ax0.legend([r,b,t2d,t3d],
                                  ['2D PIC, $T_e=0.5$ keV','2D PIC, $T_e=5$ keV','2D, kinetic','3D, kinetic'],
                                  loc=2,prop=prop,frameon=False,bbox_to_anchor=(-0.03, 1.03))#,numpoints=1)

        ax0.set_xlabel("$M_0$")
        ax0.set_ylabel("$\\Delta y $ ($\mu$m)")
        ax0.set_xlim(np.min(M),np.max(M))
        ax0.set_ylim(0,2.5)
        ax0.set_title('(a) $\delta n/n_0=0.062$')
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
    ###########
    # %% Figure b
    ###########
    Z,A = 6.,12.
    Te=500.
    Ti= Z*Te 
    cs = np.sqrt((Z*Te+3*Ti)/511000./1836./A)*c
    dnsn=0.1
    vp = -M*cs
    H2,H3 = 1./(1+Lx**2/(4*zc**2))**0.5 , np.arctan(Lx/(2*zc))/ (Lx/(2*zc))
    #cinetique
    beta2d=-np.imag(alpha_kin(Te=Te, Ti=[Ti],Z=[Z],A=[A],nisne=[1./Z], vphi=vp,figure=None,k2lde2=0) )
    Dyk2d= nesnc*dnsn*beta2d *np.sqrt(2./np.pi)* Lx/(2*sigma)*H2*Lx
    
    beta3d=0*M
    theta=np.linspace(0,np.pi/2.,10000)
    for im in range(len(M)):
        vp = -M[im]*cs*np.cos(theta)
        #a2df = np.imag( cs**2/(cs**2-vp**2 +2j*vp*cs*g0) )
        #a2df = -2*vp*cs*g0 *cs**2/ ( (cs**2-  M[im]**2*cs**2*np.cos(theta)**2 )**2 + 4*vp**2*g0**2*cs**2 )
        #beta3df[im]=np.trapz(a2df*np.cos(theta),theta)/2**.5
        a3d = -np.imag(alpha_kin(Te=Te, Ti=[Ti],Z=[Z],A=[A],nisne=[1./Z], vphi=vp,figure=None,k2lde2=0) )
        beta3d[im] =np.trapz(a3d*np.cos(theta),theta)/2**0.5
    #Dyf2d= nesnc*dnsn*beta2df *np.sqrt(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    #Dyf3d= nesnc*dnsn*beta3df *np.sqrt(2./np.pi)* Lx/(2*sigma) / np.sqrt(1+(Lx/(2*zc))**2)*Lx
    Dyk3d= nesnc*dnsn*beta3d *np.sqrt(2./np.pi)*  Lx/(2*sigma)*H3*Lx
    #PIC
    Mb =np.array([0.5, 0.9,1.2,1.5])/2**0.5
    Dyb=np.array([3, 3.7,4.8,4.9])
    Mr =np.array([0.5, 0.9,1.2,1.5])/2**0.5
    Dyr=np.array([3,5.8,5.7,5.05])
    
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        t2d, = ax0.plot(M,Dyk2d*1e6,'k',linewidth=2 )
        t3d, = ax0.plot(M,Dyk3d*1e6,'--k',linewidth=2 )
        r,   = ax0.plot(Mr,Dyr, 'or',ms=10)
        b,   = ax0.plot(Mb,Dyb, 'vb',ms=10)
        prop = fm.FontProperties(size=20)
        #first_legend = ax0.legend([t2d,t2df,t3df],['2D, kinetic','2D fluid','3D fluid'],loc=8,prop=prop)#,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        #plt.legend([t3d,t3dnl],['Collisionless', 'Non local correction'],loc=1, prop=prop,bbox_to_anchor=(1.05, 1.14))
        first_legend = ax0.legend([r,b,t2d,t3d],
                                  ['2D PIC, $T_e=0.5$ keV','2D PIC, $T_e=5$ keV','2D, kinetic','3D, kinetic'],
                                  loc=2,prop=prop,frameon=False,bbox_to_anchor=(-0.03, 1.03))#,numpoints=1)


        ax0.set_xlabel("$M_0$")
        ax0.set_ylabel("$\\Delta y $ ($\mu$m)")
        ax0.set_xlim(np.min(M),np.max(M))
        ax0.set_ylim(0,8)
        ax0.set_title('(b) $\delta n/n_0=0.2$')
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
    #############3
    # %% Figure c
    #############
    fact = 0.25*0.1*np.sqrt(np.pi/2)
    Z1,A1,Te1,Ti1=2.,4.,1000.,241.
    cs1 = np.sqrt(( Z1*Te1+3*Ti1  )/(A1*1836*511000))*3e8
    zeta1 = Z1*Te1 /511000./ (A1*1836*cs1**2/c**2)
    g1 = 0.1
    
    Z2,A2,Te2,Ti2=1.5,4.,2000.,100.
    Z2,A2,Te2,Ti2=1.,1.,2000.,2000/20.
    cs2 = np.sqrt(( Z2*Te2+3*Ti2  )/(A2*1836*511000))*3e8
    zeta2 = Z2*Te2 /511000./ (A2*1836*cs2**2/c**2)
    g2 = 0.01
    #g2 = 0.014
    
    theta=np.linspace(0,np.pi/2.,10000)
    rate1 =0*M
    rate2 =0*M
    ratef1=0*M
    ratef2=0*M
    for im in range(len(M)):
        vp1 = -M[im]*cs1*np.cos(theta)
        vp2 = -M[im]*cs2*np.cos(theta)
        #beta3df[im]=np.trapz(np.imag( cs**2/(cs**2-vp**2 +2j*vp*cs*g0) )*np.cos(theta),theta)/2**0.5
        a1 = -np.imag(alpha_kin(Te=Te1, Ti=[Ti1],Z=[Z1],A=[A1],nisne=[1./Z1], vphi=vp1,figure=None,k2lde2=0) )
        a2 = -np.imag(alpha_kin(Te=Te2, Ti=[Ti2],Z=[Z2],A=[A2],nisne=[1./Z2], vphi=vp2,figure=None,k2lde2=0) )
        rate1[im] = fact*np.trapz(a1*np.cos(theta),theta)/2**0.5
        rate2[im] = fact*np.trapz(a2*np.cos(theta),theta)/2**0.5
        
        af1= -np.imag( zeta1/(1-M[im]**2*np.cos(theta)**2 +2j*M[im]*g1*np.cos(theta)) )
        af2= -np.imag( zeta2/(1-M[im]**2*np.cos(theta)**2 +2j*M[im]*g2*np.cos(theta)) )
        ratef1[im] = fact*np.trapz(af1*np.cos(theta),theta)/2**0.5
        ratef2[im] = fact*np.trapz(af2*np.cos(theta),theta)/2**0.5
        
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        rf1, = ax0.plot(M,ratef1,'--k',linewidth=2 )
        rf2, = ax0.plot(M,ratef2,'--r',linewidth=2 )
        r1,   = ax0.plot(M,rate1, 'k',linewidth=2)
        r2,   = ax0.plot(M,rate2, 'r',linewidth=2)
        prop = fm.FontProperties(size=20)
        #first_legend = ax0.legend([t2d,t2df,t3df],['2D, kinetic','2D fluid','3D fluid'],loc=8,prop=prop)#,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        #plt.legend([t3d,t3dnl],['Collisionless', 'Non local correction'],loc=1, prop=prop,bbox_to_anchor=(1.05, 1.14))
        first_legend = ax0.legend([rf1,rf2,r1,r2],
                                  ['Fluid, $\gamma_0=0.1$','Fluid, $\gamma_0=0.01$','Kinetic, $\gamma_0=0.1$','Kinetic, $\gamma_0=0.01$'],
                                  loc=0,prop=prop,frameon=False)#bbox_to_anchor=(-0.03, 1.03))#,numpoints=1)
        ax0.set_xlabel("$M_0$")
        ax0.set_ylabel("$ (n_cT_ec/I_0)\sigma d_x \\theta  $")
        ax0.set_xlim(np.min(M),np.max(M))
        ax0.set_ylim(0,0.18)
        ax0.set_title('(c) Fluid vs. Kinetic')
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        
def bb_transient(t,M,Te=500., I0=3e14,Ti=3e3,Z=6.,A=12.,nesnc=0.1,nc=1e27,fnum=5.,Lx=128e-6,figure=1):
    c=3.e8
    n=np.sqrt(1-nesnc)
    vec=np.array(Z)/np.array(Z)
    cs = np.sqrt((Z*Te+3*Ti)/511000./1836./A)*c
    vd=M*cs
    w0=np.sqrt((1.6e-19)**2*nc/(8.85e-12*9.1e-31))
    k0=w0/c
    l0=2*np.pi/k0
    print( 'l0 = ',l0)
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    dnsn = I0*1e4/(4*Te*1.6e-19*nc*c*n)
    
    print('dnsn = ',dnsn)
   
    vp = -M*cs
    H2= 1./(1+Lx**2/(4*zc**2))**0.5 
    #cinetique
    beta2d=-np.imag(alpha_kin(Te=Te, Ti=[Ti],Z=[Z],A=[A],nisne=[1./Z], vphi=vp,figure=None,k2lde2=0) )
    yinf =  nesnc*dnsn*beta2d *np.sqrt(2./np.pi)* Lx/(2*sigma) *H2 * Lx
    print('yinf = ',yinf)
    #partie transitoire
    ###########
    vi = np.sqrt(Ti/(A*1836.)/511000.)*c    
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(    np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Z/(A*1836.)) *(Ti/Te/Z)**1.5)
    
    # old
    kold = 1.1*2./(sigma*np.pi**0.5)
    w  = cs*kold
    nu = g0*w
    print('nu/w old = ',nu/w)
    #a=np.sqrt(w**2-nu**2)
    a=np.sqrt(w**2-nu**2)
    fold = 1-np.exp(-nu*t)*( (np.cos(a*t)+nu/a*np.sin(a*t))*np.cos(w*t) +w/a*np.sin(a*t)*np.sin(w*t) )
    fold=np.real(fold)

    # new
    k = 2.**0.5/(sigma)
    w  = cs*k
    nu = g0*w
    print('nu,w new = ',nu/w)
   # a=np.sqrt(w**2-nu**2)
    #a=np.sqrt(w**2-nu**2)
    a=np.sqrt(w**2 - nu**2)
    f = 1-np.exp(-nu*t)*np.cos(k*vd*t)* (np.cos(a*t)+nu/a*np.sin(a*t))
    f=np.real(f)
    if figure is None:
        return {'yinf':yinf,'f':f,'fold':fold}
    else:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        fo, = ax0.plot(t*1e12,yinf*1e6*fold,'--k',linewidth=2 )
        f,   = ax0.plot(t*1e12,yinf*1e6*f, 'k',linewidth=2)
        
        prop = fm.FontProperties(size=20)
        #first_legend = ax0.legend([t2d,t2df,t3df],['2D, kinetic','2D fluid','3D fluid'],loc=8,prop=prop)#,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        #plt.legend([t3d,t3dnl],['Collisionless', 'Non local correction'],loc=1, prop=prop,bbox_to_anchor=(1.05, 1.14))
        first_legend = ax0.legend([f,fo],
                                  ['New','Old'],
                                  loc=0,prop=prop,frameon=False)#bbox_to_anchor=(-0.03, 1.03))#,numpoints=1)
        ax0.set_xlabel("$t$ (ps)")
        ax0.set_ylabel("$ \Delta y $ ($\mu$m)")
        ax0.set_xlim(np.min(t*1e12),np.max(t*1e12))
        #ax0.set_ylim(0,0.18)
        #ax0.set_title('(c) Fluid vs. Kinetic')
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        
#bb_transient(t=np.linspace(0,42e-12),M=0.63,figure=None)
#theory2d_article(figure=1)

#Plot figure 4c of beam bending part1
def Figure4c(figure=1):
     # %% Initialisation
    #initilisation
    nesnc=0.1
    Lx=128e-6
    c=3.e8
    n=np.sqrt(1-nesnc)
    l0=1e-6
    k0=2*np.pi/l0
    fnum=5.
    sigma = l0*fnum
    zc=np.pi*fnum**2*l0
    M=np.linspace(0,1.5,100)
    t=np.linspace(0,40e-12,10000)
    t2,t3=0*M,0*M
    
    ###########
    # %% Te = 5e2
    ###########
    Z,A = 6.,12.
    Te=500.
    Ti= Z*Te 
    cs = np.sqrt((Z*Te+3*Ti)/511000./1836./A)*c
    dnsn=0.031
    vi = np.sqrt(Ti/(A*1836.)/511000.)*c    
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(    np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Z/(A*1836.)) *(Ti/Te/Z)**1.5)

    k = 2.**0.5/(sigma)
    print('Te=500 eV, g0 = ',g0,', nu = ',g0*cs*k)
    for i in range(len(M)):
        vd=M[i]*cs
        w  = cs*k
        nu = g0*w
        a=np.sqrt(w**2 - nu**2)
        f = 1-np.exp(-nu*t)*np.cos(k*vd*t)* (np.cos(a*t)+nu/a*np.sin(a*t))
        t2[i] = t[np.argmin(np.abs( f-(1-np.exp(-1)) ))]*1e12
    
    ###########
    # %%  Te = 5e3
    ###########
    Z,A = 6.,12.
    Te=5000.
    Ti= Z*Te 
    cs = np.sqrt((Z*Te+3*Ti)/511000./1836./A)*c
    dnsn=0.1
    vi = np.sqrt(Ti/(A*1836.)/511000.)*c    
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(    np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Z/(A*1836.)) *(Ti/Te/Z)**1.5)
    k = 2.**0.5/(sigma)
    print('Te=5 keV, g0 = ',g0,', nu = ',g0*cs*k)

    for i in range(len(M)):
        vd=M[i]*cs
        w  = cs*k
        nu = g0*w
        a=np.sqrt(w**2 - nu**2)
        f = 1-np.exp(-nu*t)*np.cos(k*vd*t)* (np.cos(a*t)+nu/a*np.sin(a*t))
        t3[i] = t[np.argmin(np.abs( f-(1-np.exp(-1)) ))]*1e12
        
    #print('t2,t3 = ',t2,t3)
    
    ###########
    # %% PIC
    ###########
    Mb =np.array([0.5, 0.9,1.2,1.5])/2**0.5
    t0b=np.array([5, 3.8,4.,3.])
    Mr =np.array([0.5, 0.9,1.2,1.5])/2**0.5
    t0r=np.array([14,14.8,16.2,13.8])
        
    # %% plot
    ##################
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        t2d, = ax0.plot(M,t2,'r',linewidth=2 )
        t3d, = ax0.plot(M,t3,'b',linewidth=2 )
        r,   = ax0.plot(Mr,t0r, 'or',ms=10)
        b,   = ax0.plot(Mb,t0b, 'vb',ms=10)

        ax0.set_xlabel("$M_0$")
        ax0.set_ylabel("$t_0 $ (ps)")
        ax0.set_xlim(np.min(M),np.max(M))
        ax0.set_ylim(0,20)
        ax0.set_title('(c) $\delta n/n_0=0.062$')
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print 'xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
          
        
        